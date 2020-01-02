import numpy as np
import matplotlib.pyplot as plt

def run_Bayes_filter(bel,operation_seq,conv_filters):
    plot_histo(bel)
    for operation in operation_seq:
        action = get_action(operation)
        bel = action(bel,conv_filters[operation])
        plot_histo(bel)
    return bel

def plot_histo(bel):
    plt.cla()
    plt.bar(range(0,bel.shape[0]),bel,width=1.0)
    plt.axis([0,bel.shape[0]-1,0,1])
    plt.draw()
    plt.pause(1)

def Bayes_filter_update_forward(bel,conv_filter):
    size_area_close_to_boundary = 0
    for i in range(np.shape(conv_filter[0])[0]):
        if conv_filter[0][i] == 0:
            size_area_close_to_boundary += 1
    bel_body,bel_sec_last,bel_last = np.zeros(np.shape(bel)),np.zeros(np.shape(bel)),np.zeros(np.shape(bel))
    bel_body[0:0-size_area_close_to_boundary] = bel[0:0-size_area_close_to_boundary]
    bel_body = np.convolve(bel_body,conv_filter[0],mode='same')
    bel_sec_last[0-size_area_close_to_boundary] = bel[0-size_area_close_to_boundary]
    bel_sec_last = np.convolve(bel_sec_last,conv_filter[1],mode='same')
    bel_last[-1] = bel[-1]
    return np.add(np.add(bel_body,bel_sec_last),bel_last)

def Bayes_filter_update_backward(bel,conv_filter):
    bel_rev = np.flip(bel)
    bel_rev = Bayes_filter_update_forward(bel_rev,conv_filter)
    return np.flip(bel_rev)
    # return np.flip(Bayes_filter_update_forward(bel_rev,conv_filter))

def get_action(i):
    switcher = {
        0:Bayes_filter_update_forward,
        1:Bayes_filter_update_backward
    }
    return switcher.get(i,"Invalid action index")

def main():
    # initialization
    bel = np.hstack((np.zeros(10), 1, np.zeros(9)))
    print(bel)
    conv_filter_forward = np.array([[0,0,0.25,0.5,0.25], # for the body of array
                                    [0.25,0.75],         # for the second last cell
                                    [1]])                # for the last cell
    conv_filter_backward = conv_filter_forward
    operation_seq = np.hstack((np.zeros(9,dtype=int),np.ones(3,dtype=int))) # Moving forward and backward is marked as '0' and '1' respectively
    conv_filters = np.array([conv_filter_forward,
                            conv_filter_backward])
    plt.figure()
    bel = run_Bayes_filter(bel,operation_seq,conv_filters)

if __name__ == "__main__":
    main()
