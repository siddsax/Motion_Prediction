import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import copy

def rotmat2expmap(R):
    return quat2expmap( rotmat2quat(R) );

def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
    Args
        q: 1x4 quaternion
    Returns
        r: 1x3 exponential map
    Raises
        ValueError if the l2 norm of the quaternion is not close to 1
    """
    if (np.abs(np.linalg.norm(q)-1)>1e-3):
        raise(ValueError, "quat2expmap: input quaternion is not norm 1")
    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]
    r0    = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
    theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
    theta = np.mod( theta + 2*np.pi, 2*np.pi )
    if theta > np.pi:
        theta =  2 * np.pi - theta
        r0    = -r0
    r = r0 * theta

    return r

def rotmat2quat(R):
    """
    Converts a rotation matrix to a quaternion
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4
    Args
    R: 3x3 rotation matrix
    Returns
    q: 1x4 quaternion
    """
    rotdiff = R - R.T;
    r = np.zeros(3)
    r[0] = -rotdiff[1,2]
    r[1] =  rotdiff[0,2]
    r[2] = -rotdiff[0,1]
    sintheta = np.linalg.norm(r) / 2;
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps );
    costheta = (np.trace(R)-1) / 2;
    theta = np.arctan2( sintheta, costheta );
    q      = np.zeros(4)
    q[0]   = np.cos(theta/2)
    q[1:] = r0*np.sin(theta/2)
    
    return q

def rotmat2expmap(R):

    return quat2expmap( rotmat2quat(R) );

def rotmat2euler( R ):
    """
    Converts a rotation matrix to Euler angles
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1
    Args
    R: a 3x3 rotation matrix
    Returns
    eul: a 3x1 Euler angle representation of R
    """
    if R[0,2] == 1 or R[0,2] == -1:
        # special case
        E3   = 0 # set arbitrarily
        dlta = np.arctan2( R[0,1], R[0,2] )
        if R[0,2] == -1:
            E2 = np.pi/2
            E1 = E3 + dlta
        else:
            E2 = -np.pi/2
            E1 = -E3 + dlta
    else:
        E2 = -np.arcsin( R[0,2] )
        E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
        E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )
    eul = np.array([E1, E2, E3])

    return eul

def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
    Args
    r: 1x3 exponential map
    Returns
    R: 3x3 rotation matrix
    """
    theta = np.linalg.norm( r )
    r0  = np.divide( r, theta + np.finfo(np.float32).eps )
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
    r0x = r0x - r0x.T
    R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x)

    return R

def revertCoordinateSpace(channels_self,R0,T0):
    # addpaths
    channels_reconstruct = channels_self
    R_prev = R0
    T_prev = T0
    
    for ii in range(np.shape(channels_self)[0]):
        R_diff = expmap2rotmat(channels_self[ii,3:6])
        R = np.multiply(R_diff,R_prev)
        channels_reconstruct[ii,3:6] = rotmat2expmap(R)
        b = channels_self[ii,0:3]
        b = b.reshape((np.shape(b)[0],-1))
        T = T_prev + np.linalg.solve(R_prev,b)
        # # T = T_prev + np.multiply(np.power(R_diff,1)*(channels_self[ii,1:3]).T).T
        channels_reconstruct[ii,0:3] = T[:,0]#.reshape(np.shape(T)[0])
        T_prev = T
        R_prev = R
   
    return channels_reconstruct#,R_prev,T_prev
def euler_error(prediction, gt):
    N_SEQUENCE_TEST = 8
    R0 = np.eye(3);
    T0 = np.zeros((3,1));
    mean_errors = np.zeros((np.shape(prediction)[1], np.shape(prediction)[0]))
    for i in np.arange(N_SEQUENCE_TEST):
        eulerchannels_pred = revertCoordinateSpace(prediction[:,i,:],R0,T0)
        # Convert from exponential map to Euler angles
        for j in np.arange( eulerchannels_pred.shape[0] ):
            for k in np.arange(3,97,3):
               eulerchannels_pred[j,k:k+3] = rotmat2euler(expmap2rotmat( eulerchannels_pred[j,k:k+3] ))
        # The global translation (first 3 entries) and global rotation
        # (next 3 entries) are also not considered in the error, so the_key
        # are set to zero.
        # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
        gt_i=revertCoordinateSpace(np.copy(gt[:,i,:]),R0,T0)
        gt_i[:,0:6] = 0
        # Now compute the l2 error. The following is numpy port of the error
        # function provided by Ashesh Jain (in matlab), available at
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
        idx_to_use = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]

        euc_error = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt( euc_error )
        mean_errors[i,:] = euc_error
    
    mean_mean_errors = np.mean( mean_errors, 0 )
    
    return mean_mean_errors

