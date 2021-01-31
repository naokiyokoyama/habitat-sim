// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ESP_GFX_DOUBLESPHERECAMERASHADER_H_
#define ESP_GFX_DOUBLESPHERECAMERASHADER_H_

#include <Corrade/Containers/EnumSet.h>
#include <Magnum/Shaders/Generic.h>

#include "FisheyeShader.h"
#include "esp/core/esp.h"

namespace esp {
namespace gfx {
class DoubleSphereCameraShader : public FisheyeShader {
 public:
  enum : Magnum::UnsignedInt {
    /**
     * Color shader output. @ref shaders-generic "Generic output",
     * present always. Expects three- or four-component floating-point
     * or normalized buffer attachment.
     */
    ColorOutput = Magnum::Shaders::Generic3D::ColorOutput,

    // TODO
    /**
     * Object ID shader output. @ref shaders-generic "Generic output",
     * present only if @ref FisheyeShader::Flag::ObjectId is set. Expects a
     * single-component unsigned integral attachment. Writes the value
     * set in @ref setObjectId() there.
     */
    // ObjectIdOutput = Magnum::Shaders::Generic3D::ObjectIdOutput,
  };

  explicit DoubleSphereCameraShader(FisheyeShader::Flags flags = {
                                        FisheyeShader::Flag::ColorTexture});

  /**
   *  @brief Set the focal length of the fisheye camera
   *  @param focalLength, the focal length x, y directions
   *  @return Reference to self (for method chaining)
   */
  DoubleSphereCameraShader& setFocalLength(Magnum::Vector2 focalLength);
  /**
   *  @brief Set the offset of the principal point of the fisheye camera
   *  @param offset, the offset of the principal point in pixels
   *  @return Reference to self (for method chaining)
   */
  DoubleSphereCameraShader& setPrincipalPointOffset(Magnum::Vector2 offset);
  /**
   * @brief Set the alpha value in the "Double Sphere Camera" model.
   * See details in:
   * Vladyslav Usenko, Nikolaus Demmel and Daniel Cremers: The Double Sphere
   * Camera Model, The International Conference on 3D Vision (3DV), 2018
   *  @param alpha, the alpha value (0.0 <= alpha < 1.0)
   *  @return Reference to self (for method chaining)
   */
  DoubleSphereCameraShader& setAlpha(float alpha);

  /**
   * @brief Set the Xi value in the "Double Sphere Camera" model.
   * See details in:
   * Vladyslav Usenko, Nikolaus Demmel and Daniel Cremers: The Double Sphere
   * Camera Model, The International Conference on 3D Vision (3DV), 2018
   * @param xi, the Xi value
   * @return Reference to self (for method chaining)
   */
  DoubleSphereCameraShader& setXi(float xi);

  virtual ~DoubleSphereCameraShader(){};

  virtual DoubleSphereCameraShader& bindColorTexture(
      Magnum::GL::CubeMapTexture& texture) override;

  virtual DoubleSphereCameraShader& bindDepthTexture(
      Magnum::GL::CubeMapTexture& texture) override;

 protected:
  int focalLengthUniform_ = ID_UNDEFINED;
  int principalPointOffsetUniform_ = ID_UNDEFINED;
  int alphaUniform_ = ID_UNDEFINED;
  int xiUniform_ = ID_UNDEFINED;
  virtual void cacheUniforms() override;
  virtual void setTextureBindingPoints() override;
};

}  // namespace gfx
}  // namespace esp
#endif
