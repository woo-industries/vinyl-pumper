# vinyl-pumper

A simple ffmpeg/librosa utility meant to restore the *pump* to dance music ripped from vinyl.

- Filters out subsonic platter frequencies to create more headroom
- Two pass loudness normalization to reverse the vinyl mastering process
- Analyzes BPM and applies an appropriate compression
- Limits the signal to -11 LUFS
- Outputs back to AIFF 16 bit 44.1khz
- Recursively navigates folder structure and maintains it on output

![Pump](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExd29kM2dtdWdxeHlkajNoNGVtczZ3NHkyeDJvOXdkYXl1aGdtNHNwcCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pMljFgempfveLYJUVN/giphy.gif)

### Usage

`pip install`

then 

`python3 pump <input_dir> [output_dir] [options]`

### Options

--lufs          Target integrated loudness in LUFS (default: -11)
--peak          True peak ceiling in dBTP (default: -0.5)
--lra           Loudness range target (default: 7)
--jobs          Parallel workers (default: 4)
--format        Output format: aiff or wav (default: aiff)
--bit-depth     Output bit depth: 16 or 24 (default: 16)
--dry-run       Print what would be processed without writing files
--no-compress   Skip compression (only normalise + limit)
--bpm FLOAT     Override BPM for all files (skips auto-detection)