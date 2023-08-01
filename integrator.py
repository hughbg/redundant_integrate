import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import warnings

plt.rcParams['figure.figsize'] = [ 12, 12]
plt.rcParams["font.size"] = "15"

def corr_data_cov(signal_length, correlation_length):
    def correlation_function(i, j, correlation_length):
        distance = abs(i-j)

        # map the distance in relation to the correlation length to the range [0, 3). At the correlation length,
        # the correlation will be e^(-3) which is 0.05. This makes the value of the bottom left corner of the
        # correlation matrix = 0.05 if correlation length is the same as signal length

        return np.exp(-distance/(correlation_length)*3)

    if correlation_length == 0:
        cov = np.eye(signal_length)
    else:
        # Create correlation matrix
        cov = np.zeros((signal_length, signal_length))
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                cov[i, j] = correlation_function(i, j, correlation_length)
                
    return cov
                
def cov_samples(cov, num_signals=1):
    samples_r = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=num_signals)
    samples_i = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=num_signals)

    # Form complex
    samples_complex = samples_r+samples_i*1j

    return samples_complex
    
def correlated_signal(signal_length, correlation_length, num_signals=1):
    """ 
    The variances of each sample is 1. If the variances of each value in a sample is 1, then the covariance matrix
    is the correlation matrix. Therefore, work with the correlation matrix. 
    
    signal_length: as it says (int)
    correlation_length: int, 0 to signal_length
    num_signals: number of signals to get (int)
    """

    cov = corr_data_cov(signal_length, correlation_length)
    samples = cov_samples(cov, num_signals)
    
    return samples, cov


# Make sure there is only  1 place where baselines are allocated, so they can be counted
class BlAlloc:
    def __init__(self, true_signal=0j, noise_sigma=1, num_bl=1, num_freq=1, num_time=1, systematic_cfg=None):
        self.group_counter = 0
        self.bl_counter = 0
        self.values_counter = 0
        self.true_signal = true_signal
        self.num_bl = num_bl
        self.num_freq = num_freq
        self.num_time = num_time
        self.noise_sigma = noise_sigma
        self.systematic_cfg = systematic_cfg
        self.covariance_matrix = None

    
    def get_data(self):
        # A redundant group at one time
        ffts = self.true_signal + np.random.normal(size=(self.num_bl, self.num_freq, self.num_time), scale=self.noise_sigma)+ \
                    np.random.normal(size=(self.num_bl, self.num_freq, self.num_time), scale=self.noise_sigma)*1j

        if self.systematic_cfg is not None:
            if  self.systematic_cfg["axis"] == "freq":
                signal_length = self.num_freq
                correlation_length = int(self.systematic_cfg["correlation_fraction"]*signal_length)
                scale = self.systematic_cfg["scale"]
                num_signals = self.num_bl*self.num_time     # will reshape later. All signals are length nfreq 
                
                if self.covariance_matrix is None or self.covariance_matrix.shape[0] != signal_length:
                    # Make a new one
                    self.covariance_matrix = corr_data_cov(signal_length, correlation_length)
                    
                systematic_values = cov_samples(self.covariance_matrix, num_signals)
                systematic_values *= scale
                
                systematic_values = systematic_values.reshape(self.num_bl, self.num_time, self.num_freq)
                systematic_values = np.moveaxis(systematic_values, 2, 1)   
            else:
                assert False, "Axis "+ self.systematic_cfg["axis"]+" not implemented"

            ffts += systematic_values
        
        self.group_counter += self.num_time
        self.bl_counter += self.num_bl*self.group_counter
        self.values_counter += ffts.size
        
        return ffts
    
    def print(self):
        print("Num redundant groups allocated:", self.group_counter)
        print("Num bl allocated:", self.bl_counter)
        print("Num values allocated:", self.values_counter)


class Equation:
            
    def __init__(self):
        self.equation = {}
        self.num_accepted = 0
        self.num_rejected = 0
        
    def swap(self, a, b):
        tmp = b
        b = a
        a = tmp
        return a, b
        
    def append(self, left_bl, right_bl, left_set, right_set, where_conj, group):
        assert left_set in [ "set_1", "set_2"] and right_set in [ "set_1", "set_2" ], \
            "Can only specify set_1 or set_2. Got"+left_set+" "+right_set
        
        assert where_conj in [ "left", "right" ]
        
        term = (left_bl, right_bl, left_set, right_set, where_conj)
        
        if group not in self.equation:
            self.equation[group] = []
        
        # If already in
        is_in = False
        for group in self.equation:
            if term in self.equation[group]:
                #warnings.warn("Duplicate term will be ignored: "+str(term))
                is_in = True
                self.num_rejected += 1
                
        if not is_in:
            self.equation[group].append(term)
            self.num_accepted += 1
        
    def num_terms(self):
        num = 0
        for group in self.equation:
            num += len(self.equation[group])
        return num
    
    def all_terms(self):
        terms = []
        for group in self.equation:
            terms += self.equation[group]
        return terms
        
    
    def print(self, order=True, latex=False, time_interleaving=False):
        #print(self.num_accepted, "accepted", self.num_rejected, "rejected")
        
        s = ""
        for group in self.equation:
            if order:
                for i in range(len(self.equation[group])):
                    left_bl = self.equation[group][i][0]           
                    right_bl = self.equation[group][i][1]
                    left_set = self.equation[group][i][2]
                    right_set = self.equation[group][i][3]
                    where_conj = self.equation[group][i][4]
                               

                    # We want to have the smaller baseline index first so flip everything if this is not the case. CAREFUL.
                    if left_bl > right_bl: 
                        left_bl, right_bl = self.swap(left_bl, right_bl)
                        left_set, right_set = self.swap(left_set, right_set)
                        if where_conj == "left": where_conj = "right"
                        else: where_conj = "left"
                        self.equation[group][i] = (left_bl, right_bl, left_set, right_set, where_conj)
                  
                # Now order the whole list
                self.equation[group].sort(key=lambda a: a[0])
                        
            # Generate text for each term
            for term in self.equation[group]:
                left_bl = term[0]           
                right_bl = term[1]
                left_set = term[2]
                right_set = term[3]    
                where_conj = where_conj
                
                s += "b"+str(left_bl)+"_"+left_set
                if where_conj == "left": s += "^"
                s += "."
                s += "b"+str(right_bl)+"_"+right_set
                if where_conj == "right": s += "^"
                s += " + "
                
            s += "\n"

        equation = s[:-3]
        
        
        # Need to pretty print a little bit because the equation is in internal form
        if not time_interleaving:
            equation = equation.replace("_set_1", "")
            assert "set_2" not in equation
            
        equation = equation.replace("_set_1", "_{odd}")
        equation = equation.replace("_set_2", "_{even}")

        if latex:
            equation = equation.replace("^", "^\\dagger")
            equation = equation.replace(".", "\\times")   
            equation = equation.replace("b", "\\mathtt{b}")

        print(equation)
    
def r(x):
    return np.round(x, 4)     # precision is not working


class Integrator:
    
    def __init__(self, signal=0j, noise_sigma=1, cfg=None, systematic_cfg=None, fig_size=[12, 18], font_size="15"):
        self.signal = signal
        self.noise_sigma = noise_sigma
        self.fig_size = fig_size
        self.font_size = font_size   
        
        self.cfg = { 
            "allow_i_i": False,
            "allow_multiple_use": False,
            "all_combs": False,
            "both_conj": False,
            "both_time_directions": False,
            "time_interleaving": False
        }
        
        if cfg is not None:
            for key in cfg:
                self.cfg[key] = cfg[key]  
                
        if self.cfg["allow_i_i"] and not self.cfg["time_interleaving"]:
            raise RuntimeError("Allowing a baseline multiplied by itself at the SAME time")
        if self.cfg["both_time_directions"] and not self.cfg["time_interleaving"]:
            raise RuntimeError("Can't do both time directions when there is no time interleaving")
            
        self.systematic_cfg = systematic_cfg

    def average(self, ffts_set_1, ffts_set_2, pr=False, latex=False):
        # ffts_set_2 can be None if there is no time interleaving
        
        nbaselines = ffts_set_1.shape[0]
        equation = ""
        used = set()
        mixing = np.zeros((nbaselines, nbaselines), dtype=bool)

        
        allow_i_i = self.cfg["allow_i_i"] 
        allow_multiple_use = self.cfg["allow_multiple_use"]
        all_combs = self.cfg["all_combs"]
        both_conj = self.cfg["both_conj"]
        both_time_directions = self.cfg["both_time_directions"]
        time_interleaving = self.cfg["time_interleaving"]
                
        if time_interleaving and ffts_set_2 is None:
            raise RuntimeError("Call to average() with only 1 baseline_set and time_interleaving=True")
            

        # Generate mixing matrix
        for i in range(nbaselines):
            if all_combs: start = 0
            else: start = i
            for j in range(start, nbaselines, 1):

                if (i != j or allow_i_i) and (allow_multiple_use or ( i not in used and j not in used)):
                    mixing[i, j] = True

                    used.add(i)
                    used.add(j)
                    
                # Only way to do both conj is to use the same upper triangle mixing but transfer the conjugate to
                # the even vector. It can't be done using the lower triangle which flips the odd/even.

        # Do it
        
        set_2_if_avail = lambda : "set_2" if time_interleaving else "set_1"
        
        # b_odd^H M b_even
        ps = np.zeros_like(ffts_set_1[0])
        equation = Equation()
        for i in range(mixing.shape[0]):
            for j in range(mixing.shape[1]):
                if mixing[i, j]:   
                    equation.append(i, j, "set_1", set_2_if_avail(), "left", 1)     # Specifies b_odd^H M b_even

               
        # b_even^H  M^T  b_odd   transpose of above does conjugate
        if both_conj: 
            for i in range(mixing.shape[0]):
                for j in range(mixing.shape[1]):
                    if mixing[i, j]:  
                        equation.append(j, i, set_2_if_avail(), "set_1", "left", 2)
                        
        if both_time_directions:
            # Do it all over again but odd even flipped
            
            # b_even^H M b_odd
            for i in range(mixing.shape[0]):
                for j in range(mixing.shape[1]):
                    if mixing[i, j]:   
                        equation.append(i, j, "set_2", "set_1", "left", 3)


            # b_odd^H  M^T  b_even   transpose of above does conjugate
            if both_conj: 
                for i in range(mixing.shape[0]):
                    for j in range(mixing.shape[1]):
                        if mixing[i, j]:  
                            equation.append(j, i, "set_1", "set_2", "left", 4)
                  
        # There will have been checks done on consistency and for duplication. We can now do the integration.
        ps = np.zeros_like(ffts_set_1[0])
        num = 0
        for term in equation.all_terms():
            left_bl, right_bl, left_set, right_set, where_conj = term
            
            if left_set == "set_1": left_data = ffts_set_1[left_bl]
            else: left_data = ffts_set_2[left_bl]
            if right_set == "set_1": right_data = ffts_set_1[right_bl]
            else: right_data = ffts_set_2[right_bl]
                
            if where_conj == "left": left_data = np.conj(left_data)
            else: right_data = np.conj(right_data)

            ps += left_data*right_data
            num += 1
            
        ps /= num       

        if pr:
            print(mixing.astype(int))
            equation.print(latex=latex, time_interleaving=self.cfg["time_interleaving"])
            print("Num cross-powers:", num)
                  

        return ps
            
    def print_equation(self, latex=False):
        bls1 = np.zeros((4, 2), dtype=complex)    # Dummy     nbaselines x nfreq
      
        self.average(bls1, bls1, pr=True, latex=latex)  # Run with dummy data
            
        


    def test_run_plots(self, case, fiducial_bls=50, num_freq=131072, max_times=1000, time_step=100, max_baselines=200,
                       baseline_step=20, save_to=None):
                
        def stats(bl):
            return np.std(bl.real), scipy.stats.skew(bl.real), scipy.stats.kurtosis(bl.real), \
                    np.std(bl.imag), scipy.stats.skew(bl.imag), scipy.stats.kurtosis(bl.imag)
        
        plt.clf()
        plt.rcParams['figure.figsize'] = self.fig_size
        plt.rcParams["font.size"] = self.font_size
        
        time_interleaving = self.cfg["time_interleaving"]
 
        # Histogram on red group integrations
    
        nbins = 128
        a = self.integrate(fiducial_bls, num_freq=num_freq, num_time=2 if time_interleaving else 1)
        real_hist, real_edges = np.histogram(a.real, bins=nbins)
        imag_hist, imag_edges = np.histogram(a.imag, bins=nbins)
        real_bins = (real_edges[:-1]+real_edges[1:])/2
        imag_bins = (imag_edges[:-1]+imag_edges[1:])/2
        print("Statistics of group integration, 1 time")
        print("Real. Mean:", r(np.mean(a.real)), "Sigma:", r(np.std(a.real)), "Skew:", r(scipy.stats.skew(a.real)), 
              "Kurtosis:", r(scipy.stats.kurtosis(a.real)), "Bias", r(np.mean(a.real)/np.std(a.real)))
        print("Imag. Mean:", r(np.mean(a.imag)), "Sigma:", r(np.std(a.imag)), "Skew:", r(scipy.stats.skew(a.imag)), 
              "Kurtosis:", r(scipy.stats.kurtosis(a.imag)), "Bias", r(np.mean(a.imag)/np.std(a.imag)))

        plt.subplot(3, 1, 1)
        plt.title("Histogram of integrated redundant group, 50 bl, case "+str(case))
        plt.ylabel("Count")
        plt.xlabel("Real and imag noise power values")
        plt.plot(real_bins, real_hist, label="Real")
        plt.plot(imag_bins, imag_hist, label="Imag")
        #plt.plot(real_bins, real_hist_fit, "k", linewidth=0.5, label="Real gauss fit")
        #plt.plot(imag_bins, imag_hist_fit, "k", linewidth=0.5, label="Imag gauss fit")
        plt.legend()

        self.histogram = np.dstack((real_bins, real_hist, imag_bins, imag_hist))[0]

        print("Done histogram. Number of baseline times: ", str(fiducial_bls)+" x "+str(2 if time_interleaving else 1))

        # Increase bls

        values = []
        bls = [2, 5]+list(range(10, max_baselines, baseline_step))
        for nbl in bls:
            ps = self.integrate(nbl, num_freq=num_freq, num_time=2 if time_interleaving else 1)
            values.append(stats(ps))

        values = np.array(values)

        plt.subplot(3, 1, 2)
        plt.plot(bls, values[:, 1], label="Real skew")
        plt.plot(bls, values[:, 2], label="Real kurtosis")
        plt.plot(bls, values[:, 4], label="Imag skew")
        plt.plot(bls, values[:, 5], label="Imag kurtosis")
        plt.title("Increasing number of bls in a single redundant group, case "+str(case))
        plt.ylabel("Skew and Kurtosis")
        plt.xlabel("Number of baselines")
        plt.legend()

        self.increasing_bls = np.dstack((bls, values[:, 1], values[:, 2], values[:, 4], values[:, 5]))[0]

        print("Done increasing bls. Number of baseline times: ", "N_bl x "+str(2 if time_interleaving else 1))

        # Increase time

        times = np.concatenate((np.array([2, 6]), np.arange(10, max_times, time_step, dtype=int), np.array([max_times])))
        values = []                            
        for ntime in times:
            ps = self.integrate(fiducial_bls, num_freq=num_freq, num_time=ntime)
            values.append(stats(ps))

        integrated = ps    # Last one
        print("Final time integration statistics")
        print("Real. Mean:", r(np.mean(integrated.real)), "Sigma:", r(np.std(integrated.real)), 
              "Skew:", r(scipy.stats.skew(integrated.real)), 
              "Kurtosis:", r(scipy.stats.kurtosis(integrated.real)), "Bias", r(np.mean(integrated.real)/np.std(integrated.real)), "FINAL")
        print("Imag. Mean:", r(np.mean(integrated.imag)), "Sigma:", r(np.std(integrated.imag)), 
              "Skew:", r(scipy.stats.skew(integrated.imag)), 
              "Kurtosis:", r(scipy.stats.kurtosis(integrated.imag)), "Bias", r(np.mean(integrated.imag)/np.std(integrated.imag)), "FINAL")


        values = np.array(values)

        plt.subplot(3, 1, 3)

        plt.plot(times, values[:, 1], label="Real skew")
        plt.plot(times, values[:, 2], label="Real kurtosis")
        plt.plot(times, values[:, 4], label="Imag skew")
        plt.plot(times, values[:, 5], label="Imag kurtosis")
        plt.legend()
        plt.title("Increasing number of times that redundant group can be integrated over, 50 bls, case "+str(case))
        plt.ylabel("Skew and Kurtosis")
        plt.xlabel("Number of times")

        self.increasing_times = np.dstack((times, values[:, 1], values[:, 2], values[:, 4], values[:, 5]))[0]

        print("Done increasing times. Number of baseline times: ", str(fiducial_bls)+" x N_time")
        plt.tight_layout()
        
        if save_to is not None: plt.savefig(save_to)
            
    def integrate(self, num_bl, num_freq=1, num_time=1, no_time_avg=False, baseline_allocator=None):
        
        if self.cfg["time_interleaving"]:
            assert num_time > 1, "Can't do time interleaving with <= 1 time"
            assert num_time%2 == 0, "Can't do time interleaving with an odd number of times"                        
        
        if baseline_allocator is None:
            bl_alloc = BlAlloc(true_signal=self.signal, noise_sigma=self.noise_sigma, 
                               num_bl=num_bl, num_freq=num_freq, num_time=num_time, systematic_cfg=self.systematic_cfg) 
        else:
            bl_alloc = baseline_allocator
            
        red_groups = bl_alloc.get_data()

        all_ps = np.zeros((num_time//2 if self.cfg["time_interleaving"] else num_time, num_freq), dtype=complex)

        if self.cfg["time_interleaving"]:
            for i in range(num_time//2):
                all_ps[i] = self.average(red_groups[:, :, i*2], red_groups[:, :, i*2+1])
        else:
            for i in range(num_time):
                all_ps[i] = self.average(red_groups[:, :, i], None)

        if no_time_avg:
            return all_ps
        else:
            return np.average(all_ps, axis=0)  # Averaging over time, leaves 1 bl of length nfreq

            
    def track_imaginary(self, num_trials, num_bl, plot_x_limit=100, stats_only=False, save_to=None):
        #np.random.seed(84)
        signal_power = np.abs(self.signal)**2

        # Use freq axis as the axis for number of trials. ntime is only 1 or 2.
        trials = self.integrate(num_bl, num_freq=num_trials, num_time=2 if self.cfg["time_interleaving"] else 1)

        real_noise = trials.real-signal_power
        imag_noise = trials.imag
        
        std_real = np.std(real_noise)
        std_imag = np.std(imag_noise)
        
        if stats_only:
            return {
                "std_real": std_real,
                "std_imag": std_imag,
                "mean_real": np.mean(real_noise),
                "mean_imag": np.mean(imag_noise),
                "skew_real": scipy.stats.skew(real_noise),
                "skew_imag": scipy.stats.skew(imag_noise),
                "kurtosis_real": scipy.stats.kurtosis(real_noise),
                "kurtosis_imag": scipy.stats.kurtosis(imag_noise),
                "correlation": np.corrcoef(real_noise, imag_noise)[0, 1],
                "relative_bias_real": np.mean(real_noise)/std_real,
                "relative_bias_imag": np.mean(imag_noise)/std_imag
            }
        
        print({
                "std_real": std_real,
                "std_imag": std_imag,
                "mean_real": np.mean(real_noise),
                "mean_imag": np.mean(imag_noise),
                "skew_real": scipy.stats.skew(real_noise),
                "skew_imag": scipy.stats.skew(imag_noise),
                "kurtosis_real": scipy.stats.kurtosis(real_noise),
                "kurtosis_imag": scipy.stats.kurtosis(imag_noise),
                "correlation": np.corrcoef(real_noise, imag_noise)[0, 1]
            })
            

        # https://math.stackexchange.com/questions/1259383/calculating-uncertainty-in-standard-deviation
        std_real_err = std_real/np.sqrt(2*len(trials)-2)
        std_imag_err = std_imag/np.sqrt(2*len(trials)-2)
        
        print("Sigma of real noise+systematic:", np.round(std_real, 2), "+/-", np.round(std_real_err, 2), 
              "Sigma of imag noise+systematic:", np.round(std_imag, 2), "+/-", np.round(std_imag_err, 2)) 

        print("Pearson correlation between real and imaginary parts:",
              np.round( np.corrcoef(real_noise, imag_noise)[0, 1], 2))
        
        print("Real skew", scipy.stats.skew(real_noise), "Real kurtosis", scipy.stats.kurtosis(real_noise))
        print("Imag skew", scipy.stats.skew(imag_noise), "Imag kurtosis", scipy.stats.kurtosis(imag_noise))
        
        plt.clf()
        
        plt.rcParams['figure.figsize'] = self.fig_size
        plt.rcParams["font.size"] = self.font_size
        
        plt.subplot(2, 1, 1)
        plt.plot(np.full(plot_x_limit, signal_power), label="True val from signal")
        plt.plot(trials.real[:plot_x_limit], label="Re(signal + noise+systematic)")
        plt.plot(trials.imag[:plot_x_limit], label="Im(signal + noise+systematic)")
        plt.axhline(0, color="black", linewidth=0.5)
        plt.xlabel("Trial number")
        plt.ylabel("re(P), im(P)")
        #plt.title("Monte Carlo of redundant group integration with systematics")
        plt.legend()
        
        plt.subplot(2, 1, 2)
        nbins = 32
        real_hist, real_edges = np.histogram(real_noise, bins=nbins)
        imag_hist, imag_edges = np.histogram(imag_noise, bins=nbins)
        real_bins = (real_edges[:-1]+real_edges[1:])/2
        imag_bins = (imag_edges[:-1]+imag_edges[1:])/2
        print("Histogram")
        print("Real. Mean:", r(np.mean(real_noise)), "Sigma:", r(np.std(real_noise)), "Skew:", r(scipy.stats.skew(real_noise)), 
              "Kurtosis:", r(scipy.stats.kurtosis(real_noise)), "Bias", r(np.mean(real_noise)/np.std(real_noise)))
        print("Imag. Mean:", r(np.mean(imag_noise)), "Sigma:", r(np.std(imag_noise)), "Skew:", r(scipy.stats.skew(imag_noise)), 
              "Kurtosis:", r(scipy.stats.kurtosis(imag_noise)), "Bias", r(np.mean(imag_noise)/np.std(imag_noise)))

        plt.ylabel("Count")
        plt.xlabel("Real and imag parts")
        plt.plot(real_bins, real_hist, label="Real")
        plt.plot(imag_bins, imag_hist, label="Imag")
        #plt.plot(real_bins, real_hist_fit, "k", linewidth=0.5, label="Real gauss fit")
        #plt.plot(imag_bins, imag_hist_fit, "k", linewidth=0.5, label="Imag gauss fit")
        plt.legend()
        

        if save_to is not None:
            plt.savefig(save_to)
            
def tests():

    # Check including conjugate always gives 0 imaginary
    num_run = 0
    for both_time_directions in [ False, True ]:
        for time_interleaving in [ False, True ]:
            for both_conj in [ True ]:      # True
                for all_combs in [ False, True ]:
                    for allow_multiple_use in [ False, True ]:
                        for allow_i_i in [ False, True ]:
                            
                            cfg = { 
                                "allow_i_i": allow_i_i,
                                "allow_multiple_use": allow_multiple_use,
                                "all_combs": all_combs,
                                "both_conj": both_conj,
                                "time_interleaving": time_interleaving,
                                "both_time_directions" : both_time_directions
                            }

                            x = None
                            try:
                                integrator = Integrator(cfg=cfg)
                                y = integrator.integrate(10)
                                num_run += 1
                            except:
                                pass
                            
                            if x is not None:
                                assert np.abs(y.imag) < 1e-15
                                
    assert num_run > 0, "Phase 1: No tests ran correctly. Likely a code error."                 
    phase_1_num_run = num_run
    
    # Check that an integration with no noise always gives the signal power back
    signal = 10+8j
    for both_time_directions in [ False, True ]:
        for time_interleaving in [ False, True ]:
            for both_conj in [ False, True ]:
                for all_combs in [ False, True ]:
                    for allow_multiple_use in [ False, True ]:
                        for allow_i_i in [ False, True ]:
                            
                            cfg = { 
                                "allow_i_i": allow_i_i,
                                "allow_multiple_use": allow_multiple_use,
                                "all_combs": all_combs,
                                "both_conj": both_conj,
                                "time_interleaving": time_interleaving,
                                "both_time_directions" : both_time_directions
                            }

                            x = None
                            try:
                                integrator = Integrator(cfg=cfg, signal=signal, noise_sigma=0)

                                x = integrator.integrate(10, num_freq=10, num_time=10)
                                num_run += 1
                            except:
                                pass
                            
                            if x is not None:
                                error = np.max(np.abs(x-np.abs(signal)**2))

                                assert error == 0, "Error on signal return "+str(error)+"\nparams: "+str(cfg)+"\nsignal: " \
                                                        +str(np.abs(signal)**2)+"\nx: "+str(x)
 
    assert num_run > phase_1_num_run, "Phase 2: No tests ran correctly. Likely a code error."
    print("Tests passed")

class Case1(Integrator):
    def __init__(self, signal=0j):
        super().__init__(signal=signal)
        
class Case2(Integrator):
    def __init__(self, signal=0j):
        cfg = { "allow_multiple_use": True }
        super().__init__(signal=signal, cfg=cfg)

class Case3(Integrator):
    def __init__(self, signal=0j):
        cfg = { "time_interleaving": True, "allow_i_i": True }
        super().__init__(signal=signal, cfg=cfg)

class Case4(Integrator):
    def __init__(self, signal=0j):
        cfg = { "time_interleaving": True, "allow_multiple_use": True }
        super().__init__(signal=signal, cfg=cfg)

class Case5(Integrator):
    def __init__(self, signal=0j):
        cfg = { "time_interleaving": True, "allow_i_i": True, "allow_multiple_use": True, "all_combs": True }
        super().__init__(signal=signal, cfg=cfg)

class Case6(Integrator):
    def __init__(self, signal=0j):
        cfg = { "allow_multiple_use": True, "all_combs": True }
        super().__init__(signal=signal, cfg=cfg)

        
#case = Case6()
#case.signal = (10+10j)
#case.noise_sigma = 0
#print(case.run_one_integration(50, nfreq=2, ntime=100))

#case = Case2(signal=10+10j)
#case.print_equation()
#case.test_run_plots(3)
                  
#tests()


#num_bl = 10
#num_trials = 10
#case = Case2(signal=10+10j)
#case.print_equation()
#case.track_imaginary(num_trials, num_bl)
