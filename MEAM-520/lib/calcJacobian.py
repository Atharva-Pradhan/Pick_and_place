import numpy as np
from lib.calculateFK import FK
#import calculateFK

object = FK()

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))
    a = 0
    al = 0
    d = 0
    j = np.zeros((3,7))
    jv = np.zeros((3,8))
    X = np.identity(4)
    Y = np.identity(4)
    o = np.zeros((3,7))
    z = np.zeros((3,7))
    B = np.identity(4)

    ## STUDENT CODE GOES HERE
    t = np.array([q_in[0],q_in[1], q_in[2], q_in[3], q_in[4], q_in[5], q_in[6]]) 

    jointPositions, TOe = object.forward(t)

    for i in range(7):
            X = Y
            a,al,d = object.dhparams(i)
            B = object.matrixA(t[i], a, al,d)
            Y = np.matmul(X,B)
            o[:,i] = Y[0:3,3]
            z[:,i] = Y[0:3,2]
            #print(Y)

    o7 = Y[0:3,3]
    #print(Y)
    ozero = np.array(([0],[0],[0]))
    o = np.append(ozero,o,axis=1)
 
    zzero = np.array(([0],[0],[1]))
    z = np.append(zzero, z,axis=1)
    for i in range(7):
        j[:,i] = o7 - o[:,i]
        jv[:,i] = np.cross(z[:,i],j[:,i])
    #print(j)
    J_eight = np.append(jv,z,axis=0)
    for i in range(7):
        J[:,i] = J_eight[:,i]
    #print(J[0:3])
    return J

if __name__ == '__main__':
    q= np.array([0, 0, np.pi/2, -np.pi/2, -np.pi/4, np.pi/4, 0])
    c = calcJacobian(q)
    #print(np.round(c,3))
