import os
import sys
import argparse
from pydub import AudioSegment
import random

#noise_file_name = ['doing_the_dishes.wav','dude_miaowing.wav','exercise_bike.wav','pink_noise.wav','running_tap.wav','white_noise.wav']

def main(FLAGS):
  wav_in_file_path = FLAGS.wav_in
  wav_mix_file_path = FLAGS.wav_mix
  wav_out_file_path = FLAGS.wav_out

  if wav_in_file_path == '':
    print('please specify the wav file path --wav_in...')
    exit(0)

  if wav_out_file_path == '':
    print('please specify the csv file path --wav_out...')
    exit(0)

  #noise_file = noise_file_path + random.choice(noise_file_name)
  #print("noise_file = " + noise_file)
  
  for (dirpath, dirnames, filenames) in os.walk(wav_in_file_path):
    #file-reverse = filenames.reverse()
    for (f,mix_f) in zip(filenames,reversed(filenames)):
      if f.split(".")[-1] != "wav":
        continue
      wav_fullpath = '%s%s' % (dirpath,f) 
      wav_mix_fullpath = '%s%s' % (dirpath,mix_f)
      wav_out_fullpath = '%s%s' % (wav_out_file_path,"mix0-" + f) 
  

      input_wav = AudioSegment.from_wav(wav_fullpath)
      mix_wav = AudioSegment.from_wav(wav_mix_fullpath)

      #slice_start = random.randint(0,200)
      input_len = len(input_wav)/2
      mix_slice = mix_wav[200:700]
      #mix_slice +=5
      in_slice = input_wav[200:700]
      #in_slice -=5
      output_wav = mix_slice + in_slice
      #output_wav = input_wav.overlay(mix_slice)
      output_wav.export(wav_out_fullpath, format="wav")
	
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav_in',
      type=str,
      default='',
      help='Specify the WAV input file path.')
  parser.add_argument(
      '--wav_mix',
      type=str,
      default='',
      help='Specify the WAV Mix file path.')
  parser.add_argument(
      '--wav_out',
      type=str,
      default='',
      help='Specify the WAV output file path.')

  FLAGS, unparsed = parser.parse_known_args()
  
  main(FLAGS)
