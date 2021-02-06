import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tqdm
import math
import time
import json
import gzip
import os
import quaternion
import glob
import magnum as mn
import habitat_sim

def get_episode_info(episode_id, json_gz_path):
    with gzip.open(json_gz_path,'r') as f:
        data = f.read()

    data = json.loads(data.decode('utf-8'))
    for ep in data['episodes']:
        if int(ep['episode_id']) == int(episode_id):
            start_position = ep['start_position']
            start_quaternion = ep['start_rotation']
            scene_name = ep['scene_id']
            goal_position = ep['goals'][0]['position']
            break

    start_heading = quat_to_rad(np.quaternion(*start_quaternion))-np.pi/2

    if start_heading > np.pi:
        start_heading -= 2*np.pi
    elif start_heading < -np.pi:
        start_heading += 2*np.pi

    return start_position, start_heading, goal_position, scene_name

def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi
def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag
def quat_to_rad(rotation):
    heading_vector = quaternion_rotate_vector(
        rotation.inverse(), np.array([0, 0, -1])
    )
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi
def heading_to_quaternion(heading):
    quat = quaternion.from_euler_angles([heading+np.pi/2,0,0,0])
    quat = quaternion.as_float_array(quat)
    quat = [quat[1],-quat[3],quat[2],quat[0]]
    quat = np.quaternion(*quat)
    return mn.Quaternion(quat.imag, quat.real)

class PointHeading:
    def __init__(self, point, heading=0.):
        self.update(point, heading)

    def update(self, point, heading=None):
        self.x = point[0]
        self.y = point[2]
        self.z = point[1]
        self.point = point
        if heading is not None:
            self.heading = heading

    def __key(self):
        return (self.x, self.y, self.z, self.heading)

    def _str_key(self):
        return '{}_{}_{}_{}'.format(self.x, self.y, self.z, self.heading)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, PointHeading):
            return self.__key() == other.__key()
        return NotImplemented

class RRTStarUnicycle:
    def __init__(
        self,
        pathfinder,
        max_linear_velocity,
        max_angular_velocity,
        near_threshold,
        max_distance,
        goal_minimum_distance=0.2,
        critical_angle_lookup=None,
        directory=None
    ):
        assert max_distance<=near_threshold, (
            'near_threshold ({}) must be greater than or equal to max_distance ({})'.format(max_distance, near_threshold)
        )
        self._pathfinder     = pathfinder
        self._near_threshold = near_threshold
        self._max_distance   = max_distance
        self._max_linear_velocity  = max_linear_velocity
        self._max_angular_velocity = max_angular_velocity
        self._goal_minimum_distance = goal_minimum_distance
        self._directory = directory

        if critical_angle_lookup is None:
            self._critical_angle_lookup = self._generate_critical_angle_lookup()
        else:
            self._critical_angle_lookup = critical_angle_lookup
        
        if self._directory is not None:
            self._vis_dir  = os.path.join(self._directory, 'visualizations')
            self._json_dir = os.path.join(self._directory, 'tree_jsons')
            for i in [self._vis_dir, self._json_dir]:
                if not os.path.isdir(i):
                    os.makedirs(i)

        # TODO: Search the tree with binary search
        self.tree = {}
        self._best_goal_node = None
        self._top_down_img = None
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True
        self._start_iteration = 0
        self._start = None
        self._goal = None

    def _euclid_2D(self, p1, p2):
        return np.sqrt( (p1.x-p2.x)**2 + (p1.y-p2.y)**2 )

    def _critical_angle(self, euclid_dist, theta=np.pi/2.):
        theta_pivot = 0
        best_path_time = float('inf')
        critical_angle = 0

        while theta >= theta_pivot:
            theta_arc = theta - theta_pivot
            if np.sqrt(1-np.cos(2*theta_arc)) < 1e-6:
                arc_length = euclid_dist
            else:
                arc_length = np.sqrt(2)*euclid_dist*theta_arc/np.sqrt(1-np.cos(2*theta_arc))
            arc_time = arc_length/self._max_linear_velocity

            max_arc_turn = arc_time*self._max_angular_velocity

            if max_arc_turn > 2*theta_arc:
                pivot_time = theta_pivot/self._max_angular_velocity
                path_time = arc_time+pivot_time

                if path_time < best_path_time:
                    best_path_time = path_time
                    critical_angle = theta_arc

            theta_pivot += np.pi/180.*0.01

        return critical_angle

    def _generate_critical_angle_lookup(self, step_increment=0.1, precision=1):
        critical_angle_lookup = {}
        last_theta = np.pi/2.
        step = 0.0
        while step <= self._near_threshold:
            step = round(step, precision)
            critical_angle_lookup[step] = self._critical_angle(step)
            step += step_increment
        return critical_angle_lookup

    def _fastest_delta_heading_time(self, theta, euclid_dist, precision=1):
        critical_angle = self._critical_angle_lookup[round(euclid_dist, precision)]

        # How many degrees the robot pivots first
        theta_pivot = max(0, theta-critical_angle)
        pivot_time = theta_pivot/self._max_angular_velocity

        # The difference between the robot's heading and the destination when it starts
        # moving in an arc
        theta_arc = min(critical_angle, theta)
        arc_length = np.sqrt(2)*euclid_dist*theta_arc/np.sqrt(1-np.cos(2*theta_arc))
        arc_time = arc_length/self._max_linear_velocity

        delta_path_time = arc_time+pivot_time
        delta_heading = theta_pivot+theta_arc*2

        return delta_heading, delta_path_time

    def _get_heading_error(self, init, final):
        if init > np.pi or init < -np.pi or final > np.pi or final < -np.pi:
            raise Exception("out of range {} {}".format(init, final))
        diff = final - init
        absDiff = abs(diff)
        if absDiff == np.pi:
            return absDiff
        elif absDiff < np.pi:
            return diff
        elif final > np.pi:
            return absDiff - np.pi*2
        
        return np.pi*2 - absDiff

    def _get_path_to_start(self, pt):
        path = [pt]
        while pt != self._start:
            path.insert(0, self.tree[pt])
            pt = self.tree[pt]
        return path

    def _cost(self, pt, new_pt=None):
        if new_pt is None and pt == self._start:
            return 0

        path = self._get_path_to_start(pt)
        if new_pt is not None:
            path.append(new_pt)

        total_cost = 0
        pt = path[0]
        for new_pt in path[1:]:
            # theta is the angle from pt to new_pt (0 rad is east)
            theta = math.atan2((new_pt.y-pt.y), new_pt.x-pt.x)
            # theta_diff is angle between the robot's heading at pt to new_pt
            theta_diff = self._get_heading_error(pt.heading, theta)
            euclid_dist = self._euclid_2D(pt, new_pt)

            delta_heading, delta_path_time = self._fastest_delta_heading_time(abs(theta_diff), euclid_dist)
            if theta_diff < 0:
                final_heading = pt.heading-delta_heading
            else:
                final_heading = pt.heading+delta_heading
            if final_heading > np.pi:
                final_heading -= np.pi*2
            elif final_heading < -np.pi:
                final_heading += np.pi*2

            total_cost += delta_path_time
            pt = new_pt

        return total_cost, final_heading

    def _max_point(self, p1, p2):
        euclid_dist = self._euclid_2D(p1, p2)
        if euclid_dist <= self._max_distance:
            return p2, False

        p2.x = p1.x+(p2.x-p1.x*self._max_distance/euclid_dist)
        p2.y = p1.y+(p2.y-p1.y*self._max_distance/euclid_dist)

        return p2, True

    # This will mutate the input p
    def _closest_tree_pt(self, p):
        min_dist = float('inf')
        for tree_p in self.tree.keys():
            euclid_dist = self._euclid_2D(p, tree_p)
            if euclid_dist < min_dist:
                closest_pt = p
                min_dist = euclid_dist  

        return closest_pt

    def _path_exists(self, a, b):
        c = self._pathfinder.try_step_no_sliding(a.point, b.point)
        return np.allclose(b.point, c)

    def calculate_intermediate_points_from_start(self, end_node):
        path = self._get_path_to_start(end_node)
        self.vel_control.linear_velocity = np.array([0.0, 0.0, -self._max_linear_velocity])
        all_pts = [self._start]
        for i in range(len(path)-1):
            pt       = path[i]
            new_pt   = path[i+1]

            # theta is the angle from pt to new_pt (0 rad is east)
            theta = math.atan2((new_pt.y-pt.y), new_pt.x-pt.x)
            # theta becomes the angle between the robot's heading at pt to new_pt
            theta_diff = self._get_heading_error(pt.heading, theta)
            theta = abs(theta_diff)
            euclid_dist = self._euclid_2D(pt, new_pt)

            critical_angle = self._critical_angle_lookup[round(euclid_dist, 1)]

            # How many degrees the robot pivots first
            theta_pivot = max(0, theta-critical_angle)
            pivot_time = theta_pivot/self._max_angular_velocity

            '''
            theta_arc is the difference between the robot's heading and the destination 
            when it starts moving in an arc. Formula for arc_length dervied with trigonometry.
            Using trigonometry, we can also prove that the angle between its final heading 
            and the line connecting the start point with the end point is 2*theta_arc.
            '''
            theta_arc = min(critical_angle, theta)
            arc_length = np.sqrt(2)*euclid_dist*theta_arc/np.sqrt(1-np.cos(2*theta_arc)) # trigonometry
            arc_time = arc_length/self._max_linear_velocity
            arc_angular_vel = theta_arc*2 / arc_time 

            '''
            Determine directions for turning
            '''
            if theta_diff < 0:
                theta_pivot = -theta_pivot
                arc_angular_vel = -arc_angular_vel 

            '''
            We are only interested in the arc_time, because the robot doesn't move
            when pivoting. Get the pivoting out of the way by simple addition.
            '''
            pt_pos = pt.point
            pt_pos = np.array([pt_pos[0], self._start.z+0.5, pt_pos[2]])
            pt_quaternion = heading_to_quaternion(pt.heading+theta_pivot)
            rigid_state = habitat_sim.bindings.RigidState(pt_quaternion, pt_pos)
            self.vel_control.angular_velocity = np.array([0., arc_angular_vel, 0.])
            
            num_points = int(round(arc_length/0.1)) # TODO: Make this adjustable. Right now, every 0.1m.
            time_step = arc_time/float(num_points)
            for i in range(num_points):
                rigid_state = self.vel_control.integrate_transform(
                    time_step, rigid_state
                )

                end_heading = quat_to_rad(
                    np.quaternion(rigid_state.rotation.scalar, 
                    *rigid_state.rotation.vector)
                ) - np.pi/2
                end_pt = PointHeading(rigid_state.translation, end_heading)
                all_pts.append(end_pt)

        return all_pts

    def _str_to_pt(self, pt_str):
        x,y,z,heading = pt_str.split('_')
        point = (float(x), float(z), float(y))
        return PointHeading(point, float(heading))

    def _load_tree_from_json(self, json_path=None):
        '''
        Attempts to recover the latest existing tree from the json directory,
        or provided json_path.
        '''
        if json_path is None: 
            existing_jsons = glob.glob(os.path.join(self._json_dir, '*.json'))
            if len(existing_jsons) > 0:
                get_num_iterations = lambda x: int( os.path.basename(x).split('_')[0] )
                json_path = sorted(existing_jsons, key=get_num_iterations)[-1]
            else:
                return None

        with open(json_path) as f:
            latest_string_tree = json.load(f)

        start_str = latest_string_tree['start']
        if self._start is None:
            self._start = self._str_to_pt(start_str)

        goal_str = latest_string_tree['goal']
        if self._goal is None:
            self._goal = self._str_to_pt(goal_str)

        for k,v in latest_string_tree['graph'].items():
            pt = self._str_to_pt(k)
            if k == latest_string_tree['best_goal_node']:
                self._best_goal_node = pt

            if v == '': # start node is key
                continue
            if v == start_str:
                self.tree[pt] = self._start
            else:
                pt_v = self._str_to_pt(v)
                self.tree[pt] = pt_v
        
        self._start_iteration = int(os.path.basename(json_path).split('_')[0]) + 1

        return None

    def _visualize_tree(self, meters_per_pixel=0.01, show=False, save_path=None):
        '''
        Save and/or visualize the current tree and the best path found so far
        '''
        if self._top_down_img is None:
            self._top_down_img = self.generate_topdown_img(meters_per_pixel=meters_per_pixel)
            
            # Crop image to just valid points
            mask = cv2.cvtColor(self._top_down_img, cv2.COLOR_BGR2GRAY)
            mask[mask==255] = 0
            x,y,w,h = cv2.boundingRect(mask)
            self._top_down_img = self._top_down_img[y:y+h,x:x+w]
            
            # Determine scaling needed
            x_min, y_min = float('inf'), float('inf')
            for v in self._pathfinder.build_navmesh_vertices():
                pt = PointHeading(v)
                # Make sure it's on the same elevation as the start point
                if abs(pt.z-self._start.z) < 0.8:
                    x_min = min(x_min, pt.x)
                    y_min = min(y_min, pt.y)
            self._scale_x = lambda x: int((x-x_min)/meters_per_pixel)
            self._scale_y = lambda y: int((y-y_min)/meters_per_pixel)

        top_down_img = self._top_down_img.copy()

        # Draw entire graph in red
        for node, node_parent in self.tree.items():
            if node_parent is None: # Start point has no parent
                continue
            cv2.line(
                top_down_img,
                (self._scale_x(node.x),        self._scale_y(node.y)),
                (self._scale_x(node_parent.x), self._scale_y(node_parent.y)),
                (0,0,255),
                1
            )

        # Draw best path to goal if it exists
        if self._best_goal_node is not None:
            best_path = self.calculate_intermediate_points_from_start(self._best_goal_node)
            for pt, next_pt in zip(best_path[:-1], best_path[1:]):
                cv2.line(
                    top_down_img,
                    (self._scale_x(pt.x),       self._scale_y(pt.y)),
                    (self._scale_x(next_pt.x),  self._scale_y(next_pt.y)),
                    (0,255,0),
                    3
                )

        # Draw start and goal points
        start_x, start_y = self._scale_x(self._start.x),  self._scale_y(self._start.y)
        cv2.circle(top_down_img, (start_x, start_y), 8, (255,0,0),   -1)
        cv2.circle(top_down_img, (self._scale_x(self._goal.x),  self._scale_y(self._goal.y)),  8, (0,255,255), -1)

        # Draw start heading
        LINE_SIZE = 10
        heading_end_pt = (int(start_x+LINE_SIZE*np.cos(self._start.heading)), int(start_y+LINE_SIZE*np.sin(self._start.heading)))
        cv2.line(
            top_down_img,
            (start_x, start_y),
            heading_end_pt,
            (0,0,0),
            3
        )

        if show:
            cv2.imshow('top_down_img', top_down_img)
            cv2.waitKey(1)

        if save_path is not None:
            cv2.imwrite(save_path, top_down_img)

    def _string_tree(self):
        """
        Return the current graph as a dictionary comprised of strings to save
        to disk.
        """
        string_graph = {}
        for k,v in self.tree.items():
            if k == self._start:
                string_graph[k._str_key()] = ''
            else:
                string_graph[k._str_key()] = v._str_key()
        
        string_tree = {
            'start': self._start._str_key(),
            'goal':  self._goal._str_key(),
        }

        if self._best_goal_node is not None:
            string_tree['best_goal_node'] = self._best_goal_node._str_key()
            string_tree['best_path_time'] = self._cost(self._best_goal_node)[0]
        else:
            string_tree['best_goal_node'] = ''
            string_tree['best_path_time'] = -1

        string_tree['graph'] = string_graph

        return string_tree

    def generate_tree(
        self,
        start_position,
        start_heading,
        goal_position,
        json_path=None,
        iterations=5e4,
        visualize_on_screen=False,
        visualize_iterations=500
    ):
        start_pt = PointHeading(start_position, heading=start_heading)
        goal_pt  = PointHeading(goal_position)
        self.tree[start_pt] = None
        self._start = start_pt
        self._goal = goal_pt
        self._load_tree_from_json(json_path=json_path)

        for iteration in tqdm.trange(int(iterations+1)):
            if iteration < self._start_iteration:
                continue

            success = False
            while not success:
                # Visualize and save tree to disk
                if iteration % visualize_iterations == 0 and self._directory != '':
                    img_path  = os.path.join(self._vis_dir,  '{}_{}.png'.format( iteration, os.path.basename(self._vis_dir)))
                    json_path = os.path.join(self._json_dir, '{}_{}.json'.format(iteration, os.path.basename(self._json_dir)))
                    self._visualize_tree(save_path=img_path, show=visualize_on_screen)
                    string_tree = self._string_tree()
                    with open(json_path, 'w') as f:
                        json.dump(string_tree, f)

                # Choose random NAVIGABLE point. Must be on same plane as episode.
                rand_pt = PointHeading(self._pathfinder.get_random_navigable_point())
                if abs(rand_pt.z-self._start.z) > 0.8:
                    continue

                # Shorten distance
                closest_pt = self._closest_tree_pt(rand_pt) # TODO: Make this fast
                rand_pt, has_changed = self._max_point(closest_pt, rand_pt)
                if has_changed and not self._pathfinder.is_navigable(rand_pt.point):
                    continue

                # Find valid neighbors
                nearby_nodes = []
                for pt in self.tree.keys(): # TODO: Make this fast
                    if (
                        self._euclid_2D(rand_pt, pt) < self._near_threshold # within distance
                        and (rand_pt.x, rand_pt.y) != (pt.x, pt.y) # not the same point again
                        and self._path_exists(pt, rand_pt) # straight path exists
                    ):
                        nearby_nodes.append(pt)
                if not nearby_nodes:
                    continue

                # Find best parent from valid neighbors        
                min_cost = float('inf')
                for idx, pt in enumerate(nearby_nodes):
                    new_cost, final_heading = self._cost(pt, new_pt=rand_pt)
                    if new_cost < min_cost:
                        min_cost = new_cost
                        best_final_heading = final_heading
                        best_parent_idx = idx
                # Sometimes there is just one nearby node whose new_cost is NaN. Continue if so.
                if min_cost == float('inf'):
                    continue

                # Add/connect new node to graph
                rand_pt.heading = best_final_heading
                try:
                    self.tree[rand_pt] = nearby_nodes.pop(best_parent_idx)
                except IndexError:
                    continue

                if (
                    self._euclid_2D(rand_pt, goal_pt) < self._goal_minimum_distance
                    and (
                        self._best_goal_node is None
                        or self._cost(rand_pt)[0] < self._cost(self._best_goal_node)[0]
                    )
                ):
                    self._best_goal_node = rand_pt

                # Rewire
                for pt in nearby_nodes:
                    if pt == start_pt:
                        continue
                    new_cost = self._cost(rand_pt, new_pt=pt)[0]
                    if new_cost < self._cost(pt)[0]:
                        self.tree[pt] = rand_pt

                success = True

    def generate_topdown_img(self, meters_per_pixel=0.01):
        y = self._start.point[1]
        topdown = self._pathfinder.get_topdown_view(meters_per_pixel, y)
        topdown_bgr = np.zeros((*topdown.shape, 3), dtype=np.uint8)
        topdown_bgr[topdown==0] = (255,255,255)
        topdown_bgr[topdown==1] = (100,100,100)
        
        return topdown_bgr

    def vis_3D(self, fill=1000, flat=False):
        x,y,z = [],[],[]
        for i in self._pathfinder.build_navmesh_vertices():
            x.append(i[2])
            y.append(i[0])
            z.append(i[1])

        offset_x = min(x)
        offset_y = min(y)
        offset_z = min(z)
        x = np.array(x)-offset_x
        y = np.array(y)-offset_y
        z = np.array(z)-offset_z

        fig = plt.figure()
        if not flat:
            ax = Axes3D(fig)
            ax.scatter(z, x, y, marker='.')
        else:
            ax = plt.gca()
            ax.scatter(x, y, marker='.')

        x,y,z = [],[],[]
        for _ in range(fill):
            i = self._pathfinder.get_random_navigable_point()
            x.append(i[2]-offset_x)
            y.append(i[0]-offset_y)
            z.append(i[1]-offset_z)
        if not flat:
            ax.scatter(z, x, y, marker='.', color='yellow')
        else:
            ax.scatter(x, y, marker='.', color='yellow')

        count1, count2 = 0,0
        for _ in range(20):
            a = self._pathfinder.get_random_navigable_point()
            b = self._pathfinder.get_random_navigable_point()
            c = self._pathfinder.try_step_no_sliding(a,b)
            if np.allclose(b, c):
                count1 += 1
                color = 'green'
            else:
                count2 += 1
                color = 'red'
            if not flat:
                ax.plot(
                    [a[2]-offset_z, b[2]-offset_z],
                    [a[0]-offset_x, b[0]-offset_x],
                    [a[1]-offset_y, b[1]-offset_y],
                    color=color
                )
            else:
                ax.plot(
                    [a[2]-offset_x, b[2]-offset_x],
                    [a[0]-offset_z, b[0]-offset_z],
                    color=color
                )
        upper_lim = max(max(z),max(x))
        ax.set_xlim(0, upper_lim)
        ax.set_ylim(0, upper_lim)
        if not flat:
            ax.set_zlim(0, 0.5)

        plt.show()
