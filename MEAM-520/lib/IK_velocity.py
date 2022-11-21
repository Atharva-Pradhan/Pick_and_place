import numpy as np
import math as m
from lib.calcJacobian import calcJacobian
from lib.FK_velocity import FK_velocity



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE

    J = calcJacobian(q_in)

    body_vel = v_in
    body_vel = np.r_[body_vel,omega_in]
    body_vel = body_vel.reshape(6,1)
    j = np.array([])
    k=0
    for i in range(6):
        if np.isnan(body_vel[i]):
            j = np.r_[j,i]
            k=k+1
    j = j.astype(int)
    #body_vel = np.nan_to_num(body_vel)
    body_vel = np.delete(body_vel,j,0)
    J = np.delete(J,j,0)
    dq2 = np.linalg.lstsq(J,body_vel,rcond=None)[0]
    #print(dq2)
    dq = dq2.reshape(7,)
    #print(np.shape(dq))
    #print(dq)
    return dq
    #return dq2

if __name__ == '__main__':
    q= np.array([0,0,0,0,0,0,0])
    dq_def = np.array([1,1,1,1,1,1,1])
    v = np.array([np.nan,0,np.nan,0,0,0,0])
    #v = FK_velocity(q,dq_def)
    v_in = v[0:3]
    omega = v[3:6]
    c = IK_velocity(q,v_in,omega)
