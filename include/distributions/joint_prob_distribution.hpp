#ifndef JOINT_PROB_DISTRIBUTION_HPP
#define JOINT_PROB_DISTRIBUTION_HPP

namespace bayesopt {

class JointDistribution
{
public: 
  matrixd mCov;
  vectord mMean;

};

} // namespace bayesopt
#endif /* JOINT_PROB_DISTRIBUTION_HPP */

