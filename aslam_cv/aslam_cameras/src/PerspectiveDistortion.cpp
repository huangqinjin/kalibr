#include <aslam/cameras/PerspectiveDistortion.hpp>
#include <sm/PropertyTree.hpp>
#include <sm/serialization_macros.hpp>

namespace aslam {
namespace cameras {

PerspectiveDistortion::PerspectiveDistortion(int c) {
  if(c != 4 && c != 5 && c != 8 && c != 12 && c != 14) {
    SM_THROW(std::runtime_error, "unsuported distortion parameters " << c << ", only support 4, 5, 8, 12 or 14");
  }
  distortion.setZero(c);
}

PerspectiveDistortion::PerspectiveDistortion() {
  distortion.setZero(8);
}

PerspectiveDistortion::PerspectiveDistortion(const sm::PropertyTree & config) {
  SM_THROW(std::runtime_error, "not implemented");
}

PerspectiveDistortion::~PerspectiveDistortion() {

}

// aslam::backend compatibility
void PerspectiveDistortion::update(const double * v) {
  for(int i = 0; i < distortion.size(); ++i) {
    distortion[i] += v[i];
  }
}

int PerspectiveDistortion::minimalDimensions() const {
  return distortion.size();
}

void PerspectiveDistortion::getParameters(Eigen::MatrixXd & S) const {
  S = distortion;
}

void PerspectiveDistortion::setParameters(const Eigen::MatrixXd & S) {
  if(S.cols() != 1 || S.rows() != distortion.size()) {
    SM_THROW(std::runtime_error, "Parameters size mismatch, need " << distortion.size() << "X1, got" << S.rows() << 'X' << S.cols());
  }
  for(int i = 0; i < distortion.size(); ++i) {
    distortion[i] = S(i, 0);
  }
}

void PerspectiveDistortion::clear() {
  distortion.setZero();
}

bool PerspectiveDistortion::isBinaryEqual(
    const PerspectiveDistortion & rhs) const {
  if(!SM_CHECKMEMBERSSAME(rhs, distortion.size())) {
    return false;
  }
  for(int i = 0; i < distortion.size(); ++i) {
    if(!SM_CHECKMEMBERSSAME(rhs, distortion[i])) {
      return false;
    }
  }
  return true;
}

Eigen::Vector2i PerspectiveDistortion::parameterSize() const {
  return Eigen::Vector2i(distortion.size(), 1);
}

PerspectiveDistortion PerspectiveDistortion::getTestDistortion() {
  double d[] = {-0.2, 0.13, 0.0005, 0.0005};
  PerspectiveDistortion rt(sizeof(d) / sizeof(d[0]));
  for(int i = 0; i < rt.distortion.size(); ++i) {
    rt.distortion[i] = d[i];
  }
  return rt;
}

}  // namespace cameras
}  // namespace aslam
