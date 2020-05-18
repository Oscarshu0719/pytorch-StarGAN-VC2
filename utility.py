import glob
import librosa
import numpy as np
import os
import pysptk
import pyworld as pw
import shutil

class Singleton(type):
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance

class CommonInfo(metaclass=Singleton):
    def __init__(self, datadir: str):
        super(CommonInfo, self).__init__()
        self.datadir = datadir
    
    @property
    def speakers(self):
        """ 
            Return current selected speakers for training.
            eg. ['SF2', 'TM1', 'SF1', 'TM2']
        """

        p = os.path.join(self.datadir, "*")
        all_sub_folder = glob.glob(p)
            
        all_speaker = [s.rsplit('/', maxsplit=1)[1] for s in all_sub_folder]
        all_speaker.sort()
        return all_speaker

speakers = CommonInfo('data/spk').speakers

class Normalizer(object):
    """
        Normalizer: convience method for fetch normalize instance.
    """

    def __init__(self, statfolderpath: str='./etc'):
        self.folderpath = statfolderpath
        self.norm_dict = self.normalizer_dict()

    def forward_process(self, x, speakername):
        mean = self.norm_dict[speakername]['mcep_mean']
        std = self.norm_dict[speakername]['mcep_std']
        mean = np.reshape(mean, [-1, 1])
        std = np.reshape(std, [-1, 1])
        x = (x - mean) / std

        return x

    def backward_process(self, x, speakername):
        mean = self.norm_dict[speakername]['mcep_mean']
        std = self.norm_dict[speakername]['mcep_std']
        mean = np.reshape(mean, [-1, 1])
        std = np.reshape(std, [-1, 1])
        x = x * std + mean

        return x

    def normalizer_dict(self):
        """
            Return all speakers normailzer parameters.
        """

        d = {}
        for one_speaker in speakers:

            p = os.path.join(self.folderpath, '*.npz')
            try:
                stat_filepath = [fn for fn in glob.glob(p) if one_speaker in fn][0]
            except:
                raise Exception('No match files.')
            t = np.load(stat_filepath)
            d[one_speaker] = t

        return d
    
    def pitch_conversion(self, f0, source_speaker, target_speaker):
        """
            Logarithm Gaussian normalization for Pitch Conversions.
        """
        
        mean_log_src = self.norm_dict[source_speaker]['log_f0s_mean']
        std_log_src = self.norm_dict[source_speaker]['log_f0s_std']

        mean_log_target = self.norm_dict[target_speaker]['log_f0s_mean']
        std_log_target = self.norm_dict[target_speaker]['log_f0s_std']

        f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

        return f0_converted
    
class GenerateStatistics(object):
    def __init__(self, folder: str ='./data/processed'):
        self.folder = folder
        self.include_dict_npz = {}

        for s in speakers:
            if not self.include_dict_npz.__contains__(s):
                self.include_dict_npz[s] = []

            for one_file in os.listdir(folder):
                if one_file.startswith(s) and one_file.endswith('npz'):
                    self.include_dict_npz[s].append(one_file)
        
    @staticmethod
    def mcep_statistics(coded_sps):
        mcep_concatenated = np.concatenate(coded_sps, axis=1)
        mcep_mean = np.mean(mcep_concatenated, axis=1, keepdims=False)
        mcep_std = np.std(mcep_concatenated, axis=1, keepdims=False)
        return mcep_mean, mcep_std

    @staticmethod
    def logf0_statistics(f0s):
        log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
        log_f0s_mean = log_f0s_concatenated.mean()
        log_f0s_std = log_f0s_concatenated.std()

        return log_f0s_mean, log_f0s_std

    def generate_stats(self, statfolder: str = 'etc'):
        """
            Generate all user's statistics used for calutate normalized.
            Step 1: generate mcep mean std.
            Step 2: generate f0 mean std.
        """

        etc_path = os.path.join(os.path.realpath('.'), statfolder)
        os.makedirs(etc_path, exist_ok=True)

        for one_speaker in self.include_dict_npz.keys():
            f0s = []
            mceps = []           
            arr01 = self.include_dict_npz[one_speaker]
            if len(arr01) == 0:
                continue
            for one_file in arr01:
                t = np.load(os.path.join(self.folder, one_file))
                f0_ = np.reshape(t['f0'], [-1, 1])

                f0s.append(f0_)
                mceps.append(t['mcep'])
            
            log_f0s_mean, log_f0s_std = self.logf0_statistics(f0s)
            mcep_mean, mcep_std = self.mcep_statistics(mceps)

            print(f'log_f0s_mean: {log_f0s_mean}, log_f0s_std: {log_f0s_std}')
            print(f'mcep_mean: {mcep_mean.shape}, mcep_std: {mcep_std.shape}')

            filename = os.path.join(etc_path, f'{one_speaker}-stats.npz')
            np.savez(filename, 
                log_f0s_mean=log_f0s_mean, log_f0s_std=log_f0s_std,
                mcep_mean=mcep_mean, mcep_std=mcep_std)

            print(f'[SAVE]: {filename}')
  
    def normalize_dataset(self):
        norm  = Normalizer()
        files = librosa.util.find_files(self.folder, ext='npy')

        for p in files:
            filename = os.path.basename(p)
            speaker = filename.split(sep='_', maxsplit=1)[0]
            mcep = np.load(p)
            mcep_normed = norm.forward_process(mcep, speaker)
            os.remove(p)
            np.save(p, mcep_normed)
            print(f'[NORM]: {p}')

def world_features(wav, sr, fft_size, dim, shiftms):
    f0, timeaxis = pw.harvest(wav, sr, frame_period=shiftms)
    sp = pw.cheaptrick(wav, f0, timeaxis, sr, fft_size=fft_size)
    ap = pw.d4c(wav, f0, timeaxis, sr, fft_size=fft_size)

    return f0, timeaxis, sp, ap

def cal_mcep(wav, sr, dim, fft_size, shiftms, alpha):
    """
        Calculate MCEPs given wav singnal.
    """

    f0, timeaxis, sp, ap = world_features(wav, sr, fft_size, dim, shiftms)
    mcep = mcep_from_spec(sp, dim, alpha)
    mcep = mcep.T

    return f0, ap, mcep

def mcep_from_spec(sp, dim, alpha):
    return pysptk.sp2mc(sp, dim, alpha)

def synthesis_from_mcep(f0, mcep, ap, sr, fftsize, shiftms, alpha, rmcep=None):
    if rmcep is not None:
        mcep = mod_power(mcep, rmcep, alpha=alpha)

    if ap.shape[1] < fftsize // 2 + 1:
        ap = pw.decode_aperiodicity(ap, sr, fftsize)

    sp = pysptk.mc2sp(mcep, alpha, fftsize)

    wav = pw.synthesize(f0, sp, ap, sr, frame_period=shiftms)

    return wav

def mod_power(cvmcep, rmcep, alpha, irlen=1024):
    if rmcep.shape != cvmcep.shape:
        raise ValueError("The shapes of the converted and \
                            reference mel-cepstrum are different: \
                            {} / {}".format(cvmcep.shape, rmcep.shape))

    cv_e = pysptk.mc2e(cvmcep, alpha=alpha, irlen=irlen)
    r_e = pysptk.mc2e(rmcep, alpha=alpha, irlen=irlen)

    dpow = np.log(r_e / cv_e) / 2

    modified_cvmcep = np.copy(cvmcep)
    modified_cvmcep[:, 0] += dpow

    return modified_cvmcep

def pad_mcep(mcep_norm, frames):
    f_len = mcep_norm.shape[1]
    if  f_len >= frames: 
        pad_length = frames - (f_len - (f_len // frames) * frames)
    elif f_len < frames:
        pad_length = frames - f_len

    mcep_norm_pad = np.hstack((mcep_norm, np.zeros((mcep_norm.shape[0], pad_length))))
    return mcep_norm_pad
