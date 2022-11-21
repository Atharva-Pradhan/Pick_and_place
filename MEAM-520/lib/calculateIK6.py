from cmath import atan, cos, sin
from operator import matmul
from re import T
import numpy as np
from math import atan2, pi, acos, sqrt,cos,sin
#from calculateFK import FK

class IK:
    """
    Solves the 6 DOF (joint 5 fixed) IK problem for panda robot arm
    """
    # offsets along x direction 
    a1 = 0 
    a2 = 0
    a3 = 0.0825
    a4 = 0.0825
    a5 = 0 
    a6 = 0.088
    a7 = 0

    # offsets along z direction 
    d1 = 0.333
    d2 = 0 
    d3 = 0.316 
    d4 = 0
    d5 = 0.384
    d6 = 0 
    d7 = 0.210
    
    # This variable is used to express an arbitrary joint angle 
    Q0 = 0.123


    def panda_ik(self, target):
        """
        Solves 6 DOF IK problem given physical target in x, y, z space
        Args:
            target: dictionary containing:
                'R': numpy array of the end effector pose relative to the robot base 
                't': numpy array of the end effector position relative to the robot base 

        Returns:
             q = nx7 numpy array of joints in radians (q5: joint 5 angle should be 0)
        """
        
        # Student's code goes in between: 
        wrist_pos, T70 = self.kin_decouple(target)
        joints_467, T76 = self.ik_pos(wrist_pos)
        joints_123 = self.ik_orient(T70, joints_467, T76) 
        #q = np.array([[joints_123, joints_467[0], 0,joints_467[1],joints_467[2]],[joints_123, joints_467[0], 0,joints_467[1],joints_467[2]]])
        q = np.append(joints_123,joints_467)
        q = np.insert(q,4,0)
        q = q.reshape(7,1)
        print(q*180/(np.pi))
        # Student's code goes in between:

        ## DO NOT EDIT THIS PART 
        # This will convert your joints output to the autograder format
        #q = self.sort_joints(q)
        ## DO NOT EDIT THIS PART
        return q

    def kin_decouple(self, target):
        """
        Performs kinematic decoupling on the panda arm to find the position of wrist center
        Args: 
            target: dictionary containing:
                'R': numpy array of the end effector pose relative to the robot base 
                't': numpy array of the end effector position relative to the robot base 

        Returns:
             wrist_pos = 3x1 numpy array of the position of the wrist center in frame 7
        """
	
        R = target.get("R")
        t = target.get("t")
        #R = np.array([[0,0,-1],[0.707,-0.707,0],[-0.707,-0.707,0]])
        #t = np.array([0.256,0,-0.6435])
        #t = t.reshape(3,1)
        Rmid = np.c_[R,t]
        O = np.array([[0,0,0,1]])
        T70 = np.r_[Rmid, O]
        Rtranspose = np.transpose(R)
        t = t.reshape(3,1)
        dist = -matmul(Rtranspose,t)
        Tinvmidu = np.c_[Rtranspose,dist]
        T07 = np.r_[Tinvmidu,O]
        R07 = T07[0:3,0:3]
        O1 = np.array([0,0,1])
        O1 = O1.reshape(3,1)
        o27 = np.array(dist + (IK.d1)*np.matmul(R07,O1))
        wrist_pos = o27
        #print(wrist_pos)
        return wrist_pos, T70

    def ik_pos(self, wrist_pos):
        """
        Solves IK position problem on the joint 4, 6, 7 
        Args: 
            wrist_pos: 3x1 numpy array of the position of the wrist center in frame 7

        Returns:
             joints_467 = nx3 numpy array of all joint angles of joint 4, 6, 7
        """
        q7 = -(np.pi-(atan2(-wrist_pos[1], wrist_pos[0])+(np.pi)/4))
        T1 = [cos(q7-(np.pi)/4), -sin(q7-(np.pi)/4), 0, 0]
        T2 = [sin(q7-(np.pi)/4),  cos(q7-(np.pi)/4), 0, 0]        
        T76 = np.array([T1, 
               T2, 
               [0,0,1,IK.d7], 
               [0,0,0,1]])        
        R76 = T76[0:3,0:3]
        #print(T76)
        o26 = matmul(R76,wrist_pos)
        o26[2] = o26[2]+0.21
        #print(o26)
        oa = np.array([IK.a6,0,0])
        oa = oa.reshape(3,1)
        o25 = o26 + oa
        #print(o25)
        l4 = sqrt(pow(IK.d5,2) + pow(IK.a4,2))
        l2 = sqrt(pow(IK.d3,2) + pow(IK.a4,2))
        #print(l2)
        #print(l4)
        theta4 = acos((((o25[0])**2) + ((o25[2])**2) - ((l4)**2) - ((l2)**2))/(2*l4*l2)) 
        q4 = theta4 + atan2(IK.d3, IK.a3) + atan2(IK.d5,IK.a3) - np.pi
        #q4=0
        #print(theta4*180/np.pi)
        #t1 = atan2(o25[2],o25[0])
        theta6 = atan2(o25[2],o25[0]) - atan2(l2*sin(theta4),(l4 + l2*cos(theta4)))
        q6 = theta6 - (np.pi)/2 + atan2(IK.a3,IK.d5)
        #q6 = np.pi-q4-atan2(IK.d3,IK.a3)-atan2(o25[2],o25[0])
        #q6=0
        joints_467 = np.array([q4,q6,q7])
        
        return joints_467, T76

    def ik_orient(self, T70, joints_467, A7):
        """
        Solves IK orientation problem on the joint 1, 2, 3
        Args: 
            R: numpy array of the end effector pose relative to the robot base 
            joints_467: nx3 numpy array of all joint angles of joint 4, 6, 7

        Returns:
            joints_123 = nx3 numpy array of all joint angles of joint 1, 2 ,3
        """
        q4 = joints_467[0]
        q6 = joints_467[1]
        q7 = joints_467[2]
        q5 = 0
        A4 = np.array([[np.cos(q4), 0, -np.sin(q4), -IK.a4*np.cos(q4)],[np.sin(q4),0,np.cos(q4),-IK.a4*np.sin(q4)], [0,-1,0,0], [0,0,0,1]])
        A5 = np.array([[np.cos(q5), 0, np.sin(q5), 0],[np.sin(q5),0,-np.cos(q5),0], [0,1,0,IK.d5], [0,0,0,1]])
        A6 = np.array([[np.cos(q6), 0, np.sin(q6), IK.a6*np.cos(q6)],[np.sin(q6),0,-np.cos(q6),IK.a6*np.sin(q6)], [0,1,0,0], [0,0,0,1]])
        Ta = matmul(A4,A5)
        Tb = matmul(Ta,A6)
        T73 = matmul(Tb,A7)
        R73 = T73[0:3,0:3]
        R73d = T73[0:3,3]
        R73d = R73d.reshape(3,1)
        R73transpose = np.transpose(R73)
        R73dmain = -matmul(R73transpose,R73d)
        T73mid = np.c_[R73transpose, R73dmain]
        O4 = [[0,0,0,1]]
        T73inv = np.r_[T73mid, O4]
        
        T30 = matmul(T70,T73inv)
        print(T30)
        
        q2 = acos(T30[2,1])
        q1 = atan2(T30[1,1],T30[0,1])
        q3 = atan2(T30[2,2],T30[2,0]) 

        joints_123 = np.array([q1, q2, q3])
        return joints_123
    
    def sort_joints(self, q, col=0):
        """
        Sort the joint angle matrix by ascending order 
        Args: 
            q: nx7 joint angle matrix 
        Returns: 
            q_as = nx7 joint angle matrix in ascending order 
        """
        if col != 7: 
            q_as = q[q[:, col].argsort()]
            for i in range(q_as.shape[0]-1):
                if (q_as[i, col] < q_as[i+1, col]):
                    # do nothing
                    pass
                else:
                    for j in range(i+1, q_as.shape[0]):
                        if q_as[i, col] < q_as[j, col]:
                            idx = j
                            break
                        elif j == q_as.shape[0]-1:
                            idx = q_as.shape[0]

                    q_as_part = self.sort_joints(q_as[i:idx, :], col+1)
                    q_as[i:idx, :] = q_as_part
        else: 
            q_as = q[q[:, -1].argsort()]
        return q_as

def main(): 
    
    # fk solution code
    fk = FK()
    from operator import matmul
import numpy as np
from math import pi
import math

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        
        pass

    def forward(self, q):
        fk = FK()
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here
        X = np.identity(4);
        Y = np.identity(4);
        A = np.identity(4);
        K = np.zeros((8,3));
        t = [0, (q[0]-pi), -q[1], q[2], (q[3]-(pi/2)+pi/2), -q[4], (q[5]-(pi/2)-pi/2), q[6]-pi/4];
        for i in range(8):
            X = Y;
            a,al,d = fk.dhparams(i);
            A = fk.matrixA(t[i], a, al,d);
            Y = np.matmul(X,A);

            for j in range(3):
                    K[i][j]= Y[j][3];

            if i==2 or i==4 or i==5 or i==6:
                x = [0.195, 0.125, 0.015, 0.051];
                L = Y;
                m = 0;
                if i==2:
                    S = [[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,x[0]],
                        [0,0,0,1]];
                    L = np.matmul(L,S)
                    for m in range(3):
                        K[i][m] = L[m][3];
                
                if i==4:
                    S = [[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,-x[1]],
                        [0,0,0,1]];
                    L = np.matmul(L,S)
                    for m in range(3):
                        K[i][m] = L[m][3];
                
                if i==5:
                    S = [[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,-x[2]],
                        [0,0,0,1]];
                    L = np.matmul(L,S)
                    for m in range(3):
                        K[i][m] = L[m][3];
                
                if i==6:
                    S = [[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,x[3]],
                        [0,0,0,1]];
                    L = np.matmul(L,S)
                    for m in range(3):
                        K[i][m] = L[m][3];

        jointPositions = K;
        T0e = Y;
        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1
    def dhparams(self, i):
        a = [0,0,0,-0.0825,0.0825,0,0.088,0];
        al = [0,-pi/2,pi/2,-pi/2,-pi/2,pi/2,pi/2,0];
        d = [0.141,0.192,0,0.316,0,-0.384,0,0.21];
        return a[i], al[i], d[i];

    def matrixA(self, t,a,al,d):
        cq = math.cos(t);
        cal = math.cos(al);
        sq = math.sin(t);
        sal = math.sin(al);
        A = [[cq, -sq*cal, sq*sal, a*cq], [sq, cq*cal, -cq*sal, a*sq], [0, sal, cal, d], [0, 0, 0, 1]];
        return A

    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()

if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    #q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    #q = np.array([0,0,0,0,0,0,0])
    #joint_positions, T0e = fk.forward(q)

    #print("Joint Positions:\n",joint_positions)
    #print("End Effector Pose:\n",T0e)


    # input joints  
    q1 = 0*pi/180
    q2 = 0*pi/180
    q3 = 0*pi/180
    q4 = -90*pi/180
    q6 = -90*pi/180
    q7 = -90*pi/180
    
    q_in  = np.array([q1, q2, q3, q4, 0, q6, q7])
    [_, T_fk] = fk.forward(q_in)

    # input of IK class
    target = {'R': T_fk[0:3, 0:3], 't': T_fk[0:3, 3]}
    ik = IK()
    #print(target)
    q = ik.panda_ik(target)
    
    # verify IK solutions 
    for i in range(q.shape[0]):
        [_, T_ik] = fk.forward(q[ :])
        print('Matrix difference = ')
        print(T_fk - T_ik)
        print()

if __name__ == '__main__':
    main()
