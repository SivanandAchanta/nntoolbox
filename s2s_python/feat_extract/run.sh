# Text Feature Extraction
# python3.5 text_feats.py --uniqphns_file ../../iiith_spss_blz17_uk/etc/uniqphns_uk.txt --ehmm_dir ../../../festival_voices/iiith_uk_blizzard17/lab/ --out_dir ../feats/tfeats/

# Audio Feature Extraction
python3.5 audio_feats.py --wav_dir ../../../../festival_voices/iiith_uk_blizzard17/wav/ --out_dir ../../feats/audio_feats/log_pow_spec --fs 16000 --frsize 0.050 --frshift 0.0125 --nfft 1024 -nfilt 60 --ncep 13

# Make Train/Val/Test .npy files


