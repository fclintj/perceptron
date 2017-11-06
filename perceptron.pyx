import numpy as np
import matplotlib.pyplot as plt

def graph_vector(a_t, min_max, data_1, data_2, title):
    plt.clf()
    y_min = (-a_t[0]-a_t[1]*min_max[0])/(a_t[2])
    y_max = (-a_t[0]-a_t[1]*min_max[1])/(a_t[2])
    graph_data(data_1,'red')
    graph_data(data_2, 'blue')

    plt.plot([min_max[0],min_max[1]],[y_min,y_max]) 

    plt.ylabel('Y Data')
    plt.xlabel('X Data')
    plt.title(title)
    plt.xlim(min_max[0:2])
    plt.ylim(min_max[2:])

    plt.show()

def graph_data(data,color):
    plt.scatter(data[:,0], data[:,1], c=color)

def main():
    data_1 = np.loadtxt("data/perceptrondat1",dtype=float, delimiter=' ')
    data_2 = np.loadtxt("data/perceptrondat2",dtype=float, delimiter=' ')

    # plt.ion() 
    fig = plt.figure() # make handle to save plot 
    
    # find max and min of data-set. 
    tmp = np.vstack((data_1,data_2))
    x_min = min(tmp[:,0])
    x_max = max(tmp[:,0])
    y_min = min(tmp[:,1])
    y_max = max(tmp[:,1])
    min_max = [x_min,x_max,y_min,y_max]

    # create test data frame
    y_i = np.hstack((np.ones((len(data_1),1)),data_1))
    tmp = np.hstack((np.ones((len(data_2),1)),data_2))
    y_i = np.vstack((y_i,(tmp)*-1))
    
    # set initial values of the "a" vector
    a_t = np.array([-.1,1,1])

    mu = 0.5 
    done = 0
    iterations = 0
    times_updated = 0

    while done == 0:
        corrected = 0
        iterations += 1
        for i in range(len(y_i-1)):
            g_x = np.matmul(a_t,y_i[i].T)
            
            if g_x < 0:
                times_updated += 1
                a_t = a_t + mu*y_i[i]
                graph_vector(a_t,min_max,data_1,data_2,"Iteration " + str(times_updated))
                plt.pause(0.1)
                corrected = 1

        if corrected == 0:
            done = 1
            print("Data is linearly separable.") 
            print("It took %d iterations to find separating line"%(iterations))
            print("It took %d updates to find separating line"%(times_updated))

        elif times_updated == 10000:
            print("Linearly separable line not identified after %d iterations."%(times_updated))
            done = 1

    graph_vector(a_t,min_max,data_1,data_2,"Final Output")
    plt.pause(5)

if __name__ == '__main__':
  main()
