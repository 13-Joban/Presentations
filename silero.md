# Silero Models
1. Silero models are pre-trained AI models for audio processing tasks. 
For eg: Silero offers pre-trained models for speech-to-text conversion.
2. They improve speech recognition, transcription, and speaker identification.
For eg: Using Silero models to transcribe an audio recording of a speech or lecture into text with higher accuracy and fewer errors.

3. These models are trained on large datasets and use deep learning techniques.
4. Silero models are available for different languages including English, Spanish, French, German, and more.

5. They can be integrated into software applications for enhanced audio processing capabilities.
For Eg: The software can be used to transcribe various types of audio content, such as lectures, interviews, podcasts, webinars, and more.



## Installation

To install a Silero model using PyTorch Hub, you can use the torch.hub.load() function with the appropriate parameters. Here's an example:

```python
import torch

# Load the Silero STT model for English
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en')

# Use the model to transcribe audio
audio, sample_rate = torchaudio.load('path/to/audio/file')
output = model(audio)

# Decode the output into text
text = decoder(output[0])
print(text)

```



## Steps followed

1. Install Miniconda (can be used to install Python packages).


[s]: https://docs.conda.io/en/latest/miniconda.html.

2. Creating a environment

 ```bash
conda create --name myenv
```
3. Activate the enviroment
 ```bash
conda activate myenv

```

### Execution of the given  Example
```python 
import torch
import zipfile
import torchaudio
from glob import glob

device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

# download a single file in any format compatible with TorchAudio
torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
                               dst ='speech_orig.wav', progress=True)
test_files = glob('speech_orig.wav')
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]),
                            device=device)

output = model(input)
for example in output:
    print(decoder(example.cpu()))

```

Install required modules 

```bash
conda install pytorch torchvision torchaudio -c pytorch

```

Install omegaconf (open-source Python library for managing configuration files).
```bash
pip install omegaconf
```
or
```bash
conda install -c conda-forge omegaconf

```
Run the code 
```bash
python3 main.py
```



