// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

/* global Module */
import {
  defaultAgentConfig,
  defaultEpisode,
  defaultResolution
} from "./defaults";
import SimEnv from "./simenv_embind";
import TopDownMap from "./topdown";
import NavigateTask from "./navigate";
import { buildConfigFromURLParameters } from "./utils";

class WebDemo {
  currentResolution = defaultResolution;
  constructor(canvasId = "canvas") {
    this.canvasId = canvasId;
  }
  initializeModules(
    agentConfig = defaultAgentConfig,
    episode = defaultEpisode,
    initializeTopDown = true
  ) {
    this.config = new Module.SimulatorConfiguration();
    this.config.scene_id = Module.scene;
    this.simenv = new SimEnv(this.config, episode, 0);

    agentConfig = this.updateAgentConfigWithSensors({ ...agentConfig });

    this.simenv.addAgent(agentConfig);

    if (initializeTopDown) {
      this.topdown = new TopDownMap(
        this.simenv.getPathFinder(),
        document.getElementById("topdown")
      );
    } else {
      this.topdown = null;
    }

    this.canvasElement = document.getElementById(this.canvasId);

    this.task = new NavigateTask(this.simenv, {
      topdown: this.topdown,
      canvas: this.canvasElement,
      semantic: document.getElementById("semantic"),
      radar: document.getElementById("radar"),
      scope: document.getElementById("scope"),
      status: document.getElementById("status")
    });

    this.task.init();
    this.task.reset();
  }

  updateAgentConfigWithSensors(agentConfig = defaultAgentConfig) {
    const sensorConfigs = [
      {
        uuid: "rgb",
        sensorType: Module.SensorType.COLOR
      },
      {
        uuid: "left_eye",
        sensorType: Module.SensorType.COLOR,
        resolution: [1024, 1024]
      },
      {
        uuid: "right_eye",
        sensorType: Module.SensorType.COLOR,
        resolution: [1024, 1024]
      },
      {
        uuid: "semantic",
        sensorType: Module.SensorType.SEMANTIC,
        channels: 1
      }
    ];

    agentConfig.sensorSpecifications = sensorConfigs;
    agentConfig = this.updateAgentConfigWithResolution(agentConfig);

    return agentConfig;
  }

  resetCanvas(resolution) {
    this.canvasElement.width = resolution.width;
    this.canvasElement.height = resolution.height;
  }

  updateAgentConfigWithResolution(agentConfig) {
    agentConfig.sensorSpecifications.forEach(sensorConfig => {
      if (sensorConfig.resolution === undefined) {
        sensorConfig.resolution = [
          this.currentResolution.height,
          this.currentResolution.width
        ];
      }
    });

    return agentConfig;
  }

  display(agentConfig = defaultAgentConfig, episode = {}) {
    const config = buildConfigFromURLParameters();
    if (config.useDefaultEpisode) {
      episode = defaultEpisode;
    }

    this.initializeModules(agentConfig, episode);

    this.task.init();
    this.task.reset();
  }
}

export default WebDemo;
