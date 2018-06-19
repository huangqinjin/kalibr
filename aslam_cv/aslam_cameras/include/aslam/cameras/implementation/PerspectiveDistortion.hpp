namespace aslam {
namespace cameras {

template<typename DERIVED_Y>
void PerspectiveDistortion::distort(
    const Eigen::MatrixBase<DERIVED_Y> & yconst) const {

  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE_OR_DYNAMIC(
      Eigen::MatrixBase<DERIVED_Y>, 2);

  Eigen::MatrixBase<DERIVED_Y> & yy =
      const_cast<Eigen::MatrixBase<DERIVED_Y> &>(yconst);
  yy.derived().resize(2);

  const double k1 = distortion[0];
  const double k2 = distortion[1];
  const double p1 = distortion[2];
  const double p2 = distortion[3];
  const double k3 = distortion[4];
  const double k4 = distortion[5];
  const double k5 = distortion[6];
  const double k6 = distortion[7];
  const double x = yy[0];
  const double y = yy[1];
  const double x2 = x * x;
  const double y2 = y * y;
  const double xy = x * y;
  const double r2 = x2 + y2;
  const double r4 = r2 * r2;
  const double r6 = r2 * r4;
  const double kr1 = 1 + k1 * r2 + k2 * r4 + k3 * r6;
  const double kr2 = 1 / (1 + k4 * r2 + k5 * r4 + k6 * r6);
  const double kr = kr1 * kr2;

  yy[0] = x * kr + 2 * p1 * xy + p2 * (r2 + 2 * x2);
  yy[1] = y * kr + 2 * p2 * xy + p1 * (r2 + 2 * y2);

}

template<typename DERIVED_Y, typename DERIVED_JY>
void PerspectiveDistortion::distort(
    const Eigen::MatrixBase<DERIVED_Y> & yconst,
    const Eigen::MatrixBase<DERIVED_JY> & outJy) const {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE_OR_DYNAMIC(
      Eigen::MatrixBase<DERIVED_Y>, 2);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE_OR_DYNAMIC(
      Eigen::MatrixBase<DERIVED_JY>, 2, 2);

  Eigen::MatrixBase<DERIVED_JY> & J =
      const_cast<Eigen::MatrixBase<DERIVED_JY> &>(outJy);
  J.derived().resize(2, 2);
  J.setZero();

  Eigen::MatrixBase<DERIVED_Y> & yy =
      const_cast<Eigen::MatrixBase<DERIVED_Y> &>(yconst);
  yy.derived().resize(2);
 
  const double k1 = distortion[0];
  const double k2 = distortion[1];
  const double p1 = distortion[2];
  const double p2 = distortion[3];
  const double k3 = distortion[4];
  const double k4 = distortion[5];
  const double k5 = distortion[6];
  const double k6 = distortion[7];
  const double x = yy[0];
  const double y = yy[1];
  const double x2 = x * x;
  const double y2 = y * y;
  const double xy = x * y;
  const double r2 = x2 + y2;
  const double r4 = r2 * r2;
  const double r6 = r2 * r4;
  const double kr1 = 1 + k1 * r2 + k2 * r4 + k3 * r6;
  const double kr2 = 1 / (1 + k4 * r2 + k5 * r4 + k6 * r6);
  const double kr = kr1 * kr2;
  
  J(0, 0) = kr + 6 * p2 * x + 2 * p1 * y + (2 * k1 * x2 + 4 * k2 * x2 * r2 + 6 * k3 * x2 * r4) * kr2 -
              (2 * k4 * x2 + 4 * k5 * x2 * r2 + 6 * k6 * x2 * r4) * kr2 * kr;
  
  J(1, 1) = kr + 6 * p1 * y + 2 * p2 * x + (2 * k1 * y2 + 4 * k2 * y2 * r2 + 6 * k3 * y2 * r4) * kr2 -
              (2 * k4 * y2 + 4 * k5 * y2 * r2 + 6 * k6 * y2 * r4) * kr2 * kr;
  
  J(0, 1) = J(1, 0) = 2 * p1 * x + 2 * p2 * y + (2 * k1 * xy + 4 * k2 * xy * r2 + 6 * k3 * xy * r4) * kr2 -
                      (2 * k4 * xy + 4 * k5 * xy * r2 + 6 * k6 * xy * r4) * kr2 * kr;   
  
  yy[0] = x * kr + 2 * p1 * xy + p2 * (r2 + 2 * x2);
  yy[1] = y * kr + 2 * p2 * xy + p1 * (r2 + 2 * y2);

}

template<typename DERIVED>
void PerspectiveDistortion::undistort(
    const Eigen::MatrixBase<DERIVED> & yconst) const {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE_OR_DYNAMIC(
      Eigen::MatrixBase<DERIVED>, 2);

  Eigen::MatrixBase<DERIVED> & y =
      const_cast<Eigen::MatrixBase<DERIVED> &>(yconst);
  y.derived().resize(2);

  Eigen::Vector2d ybar = y;
  const int n = 5;
  Eigen::Matrix2d F;

  Eigen::Vector2d y_tmp;

  for (int i = 0; i < n; i++) {

    y_tmp = ybar;

    distort(y_tmp, F);

    Eigen::Vector2d e(y - y_tmp);
    Eigen::Vector2d du = (F.transpose() * F).inverse() * F.transpose() * e;

    ybar += du;

    if (e.dot(e) < 1e-15)
      break;

  }
  y = ybar;

}

template<typename DERIVED, typename DERIVED_JY>
void PerspectiveDistortion::undistort(
    const Eigen::MatrixBase<DERIVED> & yconst,
    const Eigen::MatrixBase<DERIVED_JY> & outJy) const {

  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE_OR_DYNAMIC(
      Eigen::MatrixBase<DERIVED>, 2);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE_OR_DYNAMIC(
      Eigen::MatrixBase<DERIVED_JY>, 2, 2);

  Eigen::MatrixBase<DERIVED> & y =
      const_cast<Eigen::MatrixBase<DERIVED> &>(yconst);
  y.derived().resize(2);

  // we use f^-1 ' = ( f'(f^-1) ) '
  // with f^-1 the undistortion
  // and  f the distortion
  undistort(y);  // first get the undistorted image

  Eigen::Vector2d kp = y;
  Eigen::Matrix2d Jd;
  distort(kp, Jd);

  // now y = f^-1(y0)
  DERIVED_JY & J = const_cast<DERIVED_JY &>(outJy.derived());

  J = Jd.inverse();
 
}

template<typename DERIVED_Y, typename DERIVED_JD>
void PerspectiveDistortion::distortParameterJacobian(
    const Eigen::MatrixBase<DERIVED_Y> & imageY,
    const Eigen::MatrixBase<DERIVED_JD> & outJd) const {

  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE_OR_DYNAMIC(
      Eigen::MatrixBase<DERIVED_Y>, 2);

  Eigen::MatrixBase<DERIVED_JD> & J =
      const_cast<Eigen::MatrixBase<DERIVED_JD> &>(outJd);
  J.derived().resize(2, distortion.size());
  J.setZero();

  const double k1 = distortion[0];
  const double k2 = distortion[1];
  const double p1 = distortion[2];
  const double p2 = distortion[3];
  const double k3 = distortion[4];
  const double k4 = distortion[5];
  const double k5 = distortion[6];
  const double k6 = distortion[7];
  const double x = imageY[0];
  const double y = imageY[1];
  const double x2 = x * x;
  const double y2 = y * y;
  const double xy = x * y;
  const double r2 = x2 + y2;
  const double r4 = r2 * r2;
  const double r6 = r2 * r4;
  const double kr1 = 1 + k1 * r2 + k2 * r4 + k3 * r6;
  const double kr2 = 1 / (1 + k4 * r2 + k5 * r4 + k6 * r6);
  const double kr = kr1 * kr2;

  J(0, 0) = x * r2 * kr2;
  J(0, 1) = x * r4 * kr2;
  J(0, 2) = 2 * xy;
  J(0, 3) = r2 + 2 * x2;
  J(0, 4) = x * r6 * kr2;
  J(0, 5) = -x * r2 * kr2 * kr;
  J(0, 6) = -x * r4 * kr2 * kr;
  J(0, 7) = -x * r6 * kr2 * kr;
  J(1, 0) = y * r2 * kr2;
  J(1, 1) = y * r4 * kr2;
  J(1, 2) = r2 + 2 * y2;
  J(1, 3) = 2 * xy;
  J(1, 4) = y * r6 * kr2;
  J(1, 5) = -y * r2 * kr2 * kr;
  J(1, 6) = -y * r4 * kr2 * kr;
  J(1, 7) = -y * r6 * kr2 * kr;

}

template<class Archive>
void PerspectiveDistortion::save(Archive & ar,
                                      const unsigned int /* version */) const {
  ar << BOOST_SERIALIZATION_NVP(distortion);
}

template<class Archive>
void PerspectiveDistortion::load(Archive & ar,
                                      const unsigned int version) {
  SM_ASSERT_LE(std::runtime_error, version,
               (unsigned int) CLASS_SERIALIZATION_VERSION,
               "Unsupported serialization version");

  ar >> BOOST_SERIALIZATION_NVP(distortion);
}

}  // namespace cameras
}  // namespace aslam
