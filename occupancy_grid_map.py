import numpy as np
import matplotlib.pyplot as plt

def occupancy_mapping_1d(c, bel, measurements, position_id=0, grid_resolution=10, prob_prior=0.5, flag_plot_everyframe=False):
    # parameterize this model
    prob_front_occupy = 0.3
    prob_behind_occupy = 0.6
    update_range = 20
    bel = log_odds_ratio(bel)
    for measurement in measurements:
        update_cell_id = [u for u,v in enumerate(c) if v < (measurement+update_range)]
        
        update_cell_id_front = update_cell_id[:-int(update_range/grid_resolution)]
        for id in update_cell_id_front:
            bel[id] = bel[id] + log_odds_ratio(prob_front_occupy) - log_odds_ratio(prob_prior)
        
        update_cell_id_behind = update_cell_id[-int(update_range/grid_resolution) : ]
        for id in update_cell_id_behind:
            bel[id] = bel[id] + log_odds_ratio(prob_behind_occupy) - log_odds_ratio(prob_prior)

        if flag_plot_everyframe == True:
            plot_grid_map_1d(c,inv_log_odds_ratio(bel))

    bel = inv_log_odds_ratio(bel)
    return bel

def log_odds_ratio(x):
    return np.log(x/(1.0-x))

def inv_log_odds_ratio(x):
    return 1.0-1.0/(1.0+np.exp(x))

def plot_grid_map_1d(c,m):
    plt.clf()
    plt.plot(c,m)
    plt.pause(0.05)

def main():
    # initilize
    num_cells = 21
    grid_resolution = 10
    prob_prior = 0.5
    position_id = 0
    measurements = np.array([101, 82, 91, 112, 99, 151, 96, 85, 99, 105])
    c = np.arange(num_cells) * grid_resolution
    m = np.ones(num_cells) * prob_prior

    plt.figure()
    plt.ion()
    plt.show()

    m = occupancy_mapping_1d(c, m, measurements, position_id, grid_resolution, prob_prior)
    plot_grid_map_1d(c,m)
    plt.xlabel("x-position [cm]") 
    plt.ylabel("occupancy p(x)")
    plt.show('hold')
    
if __name__ == "__main__":
    main()