import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
import matplotlib.pyplot as plt
import random

from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
'''
from calcJacobian import calcJacobian
from calculateFK import FK
from detectCollision import detectCollision
from loadmap import loadmap
'''
fk = FK()

class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    

    def __init__(self, tol=1e-4, max_steps=500, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """
        if np.linalg.norm(current-target==0):
            att_f=np.zeros(3)
        else:
            current = current.reshape(1,3)
            target = target.reshape(1,3)
            att_f = (current-target)/np.linalg.norm((current-target))
        ## STUDENT CODE STARTS HERE

        #att_f = np.zeros((3, 1)) 

        ## END STUDENT CODE

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        eta = 100000
        rho_0 = 2.5
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """
        
        midpoint = np.array([(obstacle[0]+obstacle[3])/2, (obstacle[1]+obstacle[4])/2, (obstacle[2]+obstacle[5])/2])
        rho_q_i = current - midpoint
        grad = PotentialFieldPlanner.compute_gradient(current,unitvec,obstacle)
        rep_f = eta*(1/np.linalg.norm(rho_q_i) - 1/rho_0)*(1/pow(np.linalg.norm(rho_q_i),2))*grad
        ## STUDENT CODE STARTS HERE

        #rep_f = np.zeros((3, 1)) 
        ## END STUDENT CODE
        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        #p = np.reshape(p,(1,3))
        #p = p[0]
        #box = np.reshape(box,(1,6))
        #box = box[0]
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = p.reshape(1,3)
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)
        
        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)
        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        g_1 = np.array([0.1,0.2,0.1,0.1,0.1,0.1,0.1])
        g_2 = np.array([0.1,0.1,0.7,0.1,0.5,0.5,0.1])
        zeta = np.array([2,2,2,2,2,2,2])
        zeta = zeta.reshape(1,7)
        map_coords = np.array([1,6])
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x7 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x7 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x7 numpy array representing the force vectors on each 
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE

        joint_forces = np.zeros((3, 7))
        distance = PotentialFieldPlanner.q_distance(target,current)
        F_rep_un = np.empty((3,1))
        F_rep = np.empty((1,3))
        #F_rep = []
        F_att = np.empty((1,3))
        F_att_un = np.empty((3,1))
        #print(F_rep)
        for i in range(7):
            if distance[i]>g_1[i]:
                #print((zeta[0,i]))
                F_att_un = -zeta[0,i]*(current[:,i]-target[:,i])
                F_att_un = F_att_un.reshape(1,3)
                #print(i)
                #print("before concat if att",F_att)
                #print("a",F_att_un)
                #print("a"
                F_att = np.r_[F_att,F_att_un]
                #print("after concat if att",F_att)
                #print("b",F_att)
            else:
                F_att_un = PotentialFieldPlanner.attractive_force(target[:,i],current[:,i])
                F_att_un = F_att_un.reshape(1,3)
                #print(i)
                #print("before concat else att",F_att)
                #print("c", F_att_un)
                #print("b")
                F_att = np.r_[F_att,F_att_un]
                #print("before concat else att",F_att)
                #print("d",F_att)
            for j in range(len(obstacle)):
                if distance[i]<=g_2[i]:
                    #for j in range(len(obstacle)):
                    map_coords = obstacle[j]
                    F_rep_un = PotentialFieldPlanner.repulsive_force(map_coords, current[:,i], unitvec=np.zeros((3,1)))
                    #F_rep_un = F_rep_un.reshape(3,3,1)
                    F_rep_un = np.transpose(F_rep_un)
                    #print(i)
                    #print("before concat if",F_rep)
                    #print("al",np.shape(F_rep_un))
                    F_rep = np.r_[F_rep,F_rep_un]
                    #print("after concat if",F_rep)
                else:
                    F_rep_un = np.zeros((1,3))
                    #print(i)
                    #print("before concat else",F_rep)
                    F_rep = np.r_[F_rep,F_rep_un]
                    
                    #print("after concat if",F_rep)
                    #F_rep = np.append(F_rep,F_rep_un[i],axis=1)
        #print(F_att)
        #F_rep = F_rep[7:14,:]
        #F_att = F_att[7:14,:]
        F_att = np.delete(F_att,0,0)
        F_rep = np.delete(F_rep,0,0)
        
        F_rep = F_rep.reshape((len(obstacle),7,3),order='F')
        #print("before sum", F_rep)
        F_rep = np.sum(F_rep,axis=0)
        joint_forces = -F_rep + F_att
        #print(np.shape(F_rep))
        #print("final", F_rep)
        ## END STUDENT CODE
        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x7 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x7 numpy array representing the torques on each joint 
        """
        joint_torques = np.zeros((1, 7))
        J_tot = calcJacobian(q)
        J = J_tot[0:3,:]
        J_perjoint = np.empty((3,7))
        torques = np.empty((1,7))
        for i in range(7):
            J_perjoint = np.transpose(J[:,0:i+1])
            if i>0:
                #print(np.shape(J_perjoint))
                #print(np.shape(joint_forces))
                torques = np.matmul(J_perjoint,np.transpose(joint_forces))
            #torques = np.pad(torques[i],(0,len(q)-i),'constant')
        ## STUDENT CODE STARTS HERE      
        ## END STUDENT CODE
        
        joint_torques = np.sum(torques,1)
        #print(joint_torques)
        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE

        distance = np.zeros(7)
        if np.size(target)==7 and np.size(current)==7:
            JointPositions, TOe = fk.forward(current)
            JointPositions = JointPositions[1:8,:]
            JointPositionsf, TOef = fk.forward(target)
            JointPositionsf = JointPositionsf[1:8,:]
            for i in range(7):
                distance[i] = np.linalg.norm(JointPositions[i,:] - JointPositionsf[i,:])
        elif np.size(target)==21 and np.size(current)==21:
            for i in range(7):
                distance[i] = np.linalg.norm(current[:,i] - target[:,i])
        ## END STUDENT CODE
        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct):
        b = []
        uv = []
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task
        """
        ## STUDENT CODE STARTS HERE

        dq = np.zeros((1, 7))
        if np.size(q)==7:
            JointPos, TOe = fk.forward(q)
            for i in range(7):
                JointPos[i] = np.reshape(JointPos[i],(1,3))
                b[i], uv[i] = PotentialFieldPlanner.dist_point2box(JointPos[i,:], map_struct)
                b[i] = b[i].reshape(3,1)
                dq = (np.transpose(JointPos[i,:])-b[i])/np.linalg.norm((np.transpose(JointPos[i,:])-b[i]))
        elif np.size(q)==3:
            b, uv = PotentialFieldPlanner.dist_point2box(np.transpose(q),map_struct)
            uv = uv.reshape(3,1)
            q = q.reshape(3,1)
            dq = (q-b*uv)/np.linalg.norm(q-b*uv)
        ## END STUDENT CODEs
        return dq

    @staticmethod
    def randomwalk(q):
        q_dash=[]
        direction = [0,1]
        dir = random.choice(direction)
        m = random.uniform(1, 2)
        if dir==0:
            q_dash = q+m
            m=m+1
        else:
            q_dash = q-m
            m=m+1
        return q_dash

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        
        max_steps = 1000
        alpha = 0.0001
        k = 0
        q = start
        q_path = q
        qf = goal
        epsilon = 1e-4*np.ones(len(q))
        JP, T0e = fk.forward(q)
        JPF, T0ef = fk.forward(qf)
        jointPositions = np.delete(JP,7,0)
        jointPositionsf = np.delete(JPF,7,0)
        dist = PotentialFieldPlanner.q_distance(q,qf)
        while (dist<epsilon).all:    
            joint_forces = PotentialFieldPlanner.compute_forces(np.transpose(jointPositionsf),map_struct.obstacles,np.transpose(jointPositions))
            joint_torques = PotentialFieldPlanner.compute_torques(joint_forces,q)
            Tau = joint_torques
            a1 = q
            q = q+alpha*Tau/np.linalg.norm(Tau)
            a2 = q
            if (a1==a2).all:
                q = PotentialFieldPlanner.randomwalk(q)
            k = k+1
            #print(k)
            if k>max_steps:
                break
            #q = q.reshape(1,7)
            q_path = np.r_[q_path, q]
            #print(map_struct.obstacles[0])
            print(len(map_struct.obstacles))
            for l in range(len(map_struct.obstacles)):
                jpa1, Ta1 = fk.forward(a1)
                jpa1 = np.delete(jpa1,7,0)
                jpa2, Ta2 = fk.forward(a2)
                jpa1 = np.delete(jpa2,7,0)
                a = detectCollision(jpa1,jpa2,map_struct.obstacles[l])
                #print(a)
                a = np.reshape(a,(7,1))
                if (a).all:
                    break
                n1 = jpa1[0:6]
                n2 = jpa1[1:7]
                y = detectCollision(n1,n2,map_struct.obstacles[l])
                y = np.reshape(y,(6,1))
                if (y).all:
                    break
            
            for l in range(np.shape(q_path)[0]):
                if (q_path[l]<self.lower).any or (q_path[l,:]>self.upper).any:
                    break
        if k<max_steps:
            q_path = q_path.reshape(k+1,7)
        else:
            q_path = q_path.reshape(k,7)
        q_path = np.array(q_path)
        #print(q_path)
            ## STUDENT CODE STARTS HERE
            
            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code 
            
            # Compute gradient 
            # TODO: this is how to change your joint angles 

            # Termination Conditions
            #if True: # TODO: check termination conditions
            	#break # exit the while loop if conditions are met!

            # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES
            # TODO: Figure out how to use the provided function 

            # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
            # TODO: when detect a local minima, implement a random walk
            
            ## END STUDENT CODE
        return q_path
        
        
            
	
################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    #map_struct = loadmap("../maps/map1.txt")
    map_struct = loadmap("maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    #goal =  np.array([-1.2, -1.57 , +1.57, 2.07, -1.57, 1.57, 3.14])
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[-1], goal)
        #print('iteration:',i)#,' q =', q_path[i, :], 'goal=',  goal)#, ' error={error:3.4f}'.format(error=error))
        #print("error",error)
    #print("q path: ", q_path[0])
