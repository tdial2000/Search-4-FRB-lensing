# imports
import numpy as np 
import matplotlib.pyplot as plt 
import argparse

from matplotlib.widgets import TextBox



#########  UTILITY FUNCTIONS ###########
def circle(r, N = 360):
        x = r * np.cos(np.linspace(0, 2*np.pi, N))
        y = r * np.sin(np.linspace(0, 2*np.pi, N))
        return x, y


# create class

class point_mass_lense:

    # constants
    G = 6.67430e-11
    Msun = 1.989e30
    c = 2.998e8
    Mpc2m = 3.086e22
    arcsec2rad = 3600

    def __init__(self, M = 100.0, z = 0.1, rMin = 0.1, rMax = 5.0, N = 100):

        self.M = M
        self.z = z

        # sim params
        self.rMin = rMin
        self.rMax = rMax
        self.N = N

        # distances
        self.D_l = 400 * self.Mpc2m
        self.D_ls = .01 * self.Mpc2m
        self.D_s = self.D_l + self.D_ls

        # calculate parametric values
        # Einstein radius
        self.theta_E = np.sqrt(4 * self.G * self.M / self.c**2 * (self.D_ls / (self.D_l * self.D_s)))

        # Swarszchild radius
        self.Rs = 2 * self.G * self.M / self.c**2

        # axes
        self.ax = {}

        # widgets
        self.widgets = {}

        # info boxes
        self._info_boxes = {}
        self._info_boxes['glens_move'] = None



    def calc_delays(self, y):
        return (4 * self.G * self.M * self.Msun / self.c**3 * (1+self.z) * 
                (0.5*y*np.sqrt(y**2+4) + np.log((np.sqrt(y**2+4)+y)/(np.sqrt(y**2+4)-y))))


    def calc_magnifications(self, y):
        return (y**2 + 2 - y*np.sqrt(y**2 + 4)) / (y**2 + 2 + y*np.sqrt(y**2 + 4))



    def _plot_glens(self):

        y = np.linspace(self.rMin, self.rMax, self.N)[::-1]

        # calculate shapiro delays
        delays = self.calc_delays(y)

        # calculate magnifications
        mags = self.calc_magnifications(y)

        colors = plt.cm.inferno(np.linspace(0, 1, self.N))[::-1]

        # plot
        delays_max = np.max(delays)
        for i, yi in enumerate(y):
            col = colors[int(delays[i]/delays_max*(delays.size - 1))]
            self.ax['glens'].fill(*circle(yi), color = col, edgecolor = None)
        
        # mag ratio contours
        for i in [10, 20, 50, 100, 200, 500, 1000]:
            mags_ind = np.argmin(np.abs(1/mags - i))
            print(mags_ind)
            if mags_ind > 0:
                self.ax['glens'].plot(*circle(y[mags_ind]), color = [0.5, 0.5, 0.5], linestyle = '--')
        
        # Einstien radius contour
        self.ax['glens'].plot(*circle(1), ':', linewidth = 1.5)
        self.ax['glens'].grid(alpha = 0.4)

        self.ax['glens'].set_xlabel("X [Einstein radii]", fontsize = 16)
        self.ax['glens'].set_ylabel("Y [Einstein radii]", fontsize = 16)


        # colorbar axes
        self.ax['glens_cmap'].imshow(delays.reshape(1, delays.size), aspect = 'auto',
                                    cmap = 'inferno', extent = [delays[-1]*1e3, delays[0]*1e3, 0, 1])
        self.ax['glens_cmap'].set_xlabel("TIme delay [ms]", fontsize = 16)
        self.ax['glens_cmap'].get_yaxis().set_visible(False)




    def _get_glens_info(self, event):
        if not self.ax['glens'].in_axes(event):
            return
        
        # get info given position of mouse
        y = np.sqrt(event.xdata**2 + event.ydata**2)

        delay = self.calc_delays(y)
        mag = self.calc_magnifications(y)

        text = "".join((f"Delay [ms]: {delay*1e3:.4f}\n", 
                          f"Flux ratio: {1/mag:.4f}"))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        if self._info_boxes['glens_move'] is None:
            self._info_boxes['glens_move'] = self.ax['glens'].text(0.05, 0.95, 
                        text, transform=self.ax['glens'].transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)
        else:
            self._info_boxes['glens_move'].set_text(text)

        self.fig.canvas.draw()



    def _update_mass(self, M):
        self.M = float(M)
        self._update_lens()


    def _update_z(self, z):
        self.z = float(z)
        self._update_lens()


    def _update_lens(self):

        # delete current lens
        self.ax['glens'].clear()
        self.ax['glens_cmap'].clear()
        self._info_boxes['glens_move'] = None

        self._plot_glens()



    def plot(self):
        """
        
        Plot a bunch of infomation

        1. Shapiro delay
        2. Flux magnification ratio
        3. initialise widgets

        """

        self.fig = plt.figure(figsize = (10, 8))

        # Create face on contour of grav lens
        self.ax['glens'] = self.fig.add_axes([0.08, 0.25, 0.70, 0.70])


        # Colormap
        self.ax['glens_cmap'] = self.fig.add_axes([0.08, 0.10, 0.7, 0.05])
        self._plot_glens()



        # textboxes
        # M_L
        self.ax['ML_textbox'] = self.fig.add_axes([0.81, 0.90, 0.07, 0.02])
        self.widgets['ML_textbox'] = TextBox(self.ax['ML_textbox'], label = "$M_{L}$",  
                                                  initial = self.M, textalignment = "left", label_pad = 0.04)
        self.widgets['ML_textbox'].on_submit(self._update_mass)

        # z (redshift)
        self.ax['z_textbox'] = self.fig.add_axes([0.81, 0.85, 0.07, 0.02])
        self.widgets['z_textbox'] = TextBox(self.ax['z_textbox'], label = "z",  
                                                  initial = self.z, textalignment = "left", label_pad = 0.04)
        self.widgets['z_textbox'].on_submit(self._update_z)

        # set callback for mouse hover over glens axes
        self.fig.canvas.mpl_connect('motion_notify_event', self._get_glens_info)




        # Activate figure
        plt.show()













def get_args():


    parser = argparse.ArgumentParser()
    parser.add_argument('-M', help = "Mass of lens (in solar masses)", type = float, default = 100.0)
    parser.add_argument('-z', help = "Lens redshift", type = float, default = 0.1)

    args = parser.parse_args()

    return args





# main program 
if __name__ == "__main__":

    args = get_args()

    # make instance of lens class
    pml = point_mass_lense(M = args.M, z = args.z)

    # plot
    pml.plot()

