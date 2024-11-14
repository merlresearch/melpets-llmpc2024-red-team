<!--
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: Apache-2.0
-->

# MEL-PETs Joint-Context Attack for LLM Privacy Challenge

## Features

Code for the Joint-Context Attack submission by the MEL-PETs team for the [NeurIPS 2024 LLM Privacy Challenge](https://llm-pc.github.io/) Red Team track.

## Installation

The code depends on `jsonlines`, `tqdm`, `torch`, and `unsloth`.

The `jsonlines` and  `tqdm` packages can be installed with pip (or similarly with conda):

```sh
pip install jsonlines tqdm
```

To install `torch` and `unsloth`, we recommend following the installation guides provided by those libraries (as this should be customized based your specific CUDA setup, and may require manually installed further sub-dependencies, as instructed):

- https://pytorch.org/get-started/locally/
- https://docs.unsloth.ai/get-started/installation

### Automatic environment setup

While we recommend following the installation guides for `torch` and `unsloth`, in order to properly install those packages with all of their necessary dependencies, as tailored for your specific environment.
However, here is a rough guide for an automated alternative:

```sh
conda create --name unsloth_env python=3.11
conda activate unsloth_env
pip install -r requirements.txt
```

Note: the above `requirements.txt` does not list specific package versions and relies on pip to figure out the necessary CUDA environment automatically. If this is not working, you may need to select specific package and CUDA versions manually, as well as manually installing some sub-dependencies of these libraries (see the above installation guides for `torch` and `unsloth`).

We have tested that the code runs in the following environment, as reported by unsloth:

```
==((====))==  Unsloth 2024.11.5: Fast Llama patching. Transformers = 4.46.2.
   \\   /|    GPU: NVIDIA L40S. Max memory: 44.521 GB. Platform = Linux.
O^O/ \_/ \    Pytorch: 2.5.1+cu124. CUDA = 8.9. CUDA Toolkit = 12.4.
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.28.post3. FA2 = False]
 "-____-"     Free Apache license: http://github.com/unslothai/unsloth
```

## Usage

The code is contained within a single python file that can be run with:

```sh
python main.py
```

This runs the attack with default settings that were used for our final test submission, and should reproduce the included file [result.jsonl](./result.jsonl).

## Citation

If you use the software, please cite the following paper (note: currently under review for the competition):

```BibTeX
@inproceedings{melpets_red,
    author = {Ye Wang and Tsunato Nakai and Jing Liu and Toshiaki Koike-Akino and Kento Oonishi and Takuya Higashi},
    title = {MEL-PETs Joint-Context Attack for the NeurIPS 2024 LLM Privacy Challenge Red Team Track},
    booktitle = {NeurIPS 2024 LLM Privacy Challenge (under review)},
    year = 2024
}
```

## Related Links

- [NeurIPS 2024 LLM Privacy Challenge](https://llm-pc.github.io/)

## Contact

Ye Wang <yewang@merl.com>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `Apache-2.0` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:

```
Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
