from librosa import effects
from tqdm import tqdm
from glob import glob
import numpy as np
from scipy.io import wavfile as wf
from os.path import join as jp, basename as bn


def main():
  tta_speed = 0.5 # slow down (i.e. &lt; 1.0)
  samples_per_sec = 16000
  test_fns = sorted(glob('data/one/*.wav'))
  tta_dir = './test/audio/one'
  for fn in tqdm(test_fns):
    basename = bn(fn)
    basename = "sp5-" + basename 
    rate, data = wf.read(fn)
    assert len(data) == samples_per_sec
    data = np.float32(data) / 32767
    data = effects.time_stretch(data, tta_speed)
    data = data[-samples_per_sec:]
    out_fn = jp(tta_dir, basename)
    wf.write(out_fn, rate, np.int16(data * 32767))


if __name__ == '__main__':
  main()

