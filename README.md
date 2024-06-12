# :house_with_garden: ParaHome

> ParaHome: Parameterizing Everyday Home Activities Towards 3D Generative Modeling of Human-Object Interactions (arXiv 2024)  
> [Jeonghwan Kim\*](https://canoneod.github.io/), [Jisoo Kim\*](https://jlogkim.github.io/), [Jeonghyeon Na](https://nagooon.github.io/), [Hanbyul Joo](https://jhugestar.github.io/)  
> \[[Project Page](https://jlogkim.github.io/parahome/)\] \[[Paper](https://arxiv.org/pdf/2401.10232.pdf)\] \[[Supp. Video](https://www.youtube.com/embed/HeXqiK0eGec?si=mtAmctx0JHHYD6Ac)\]

<p align="center">    
    <img src="assets/teaser.jpg" alt="Image" width="100%"/>
</p>
This is a repository of the ParaHome system. Our system is designed to capture human-object interaction in a natural home environment. We parameterized all 3D movements of body, hands, and objects and captured large-scale dataset for human-object interaction.

### News

- 🎊 Parahome demo data is available!

### Download Demo files

```
wget http://vcl.snu.ac.kr/parahome/data.zip
```

unzip and move scan, seq directories into data directory

```
.
├── assets
├── visualize
├── data
│   ├── scan
└── └── seq
```

### Environment Setting

Check out [install.md](./install.md)

### Visualize Demo files

To visualize the demo parahome data, select sequence path in the data/seq directory and execute the command

```
cd visualize
python render.py --scene_root data/seq/s01
```

### Visualize via WebGL

To visualize via WebGL with interactive tools, execute under commands after download and unzip the data files

```
python -m http.server
```

Open under link on your browser. You can select sequence from s01 to s02.

http://localhost:8000/visualize-web/?seq=s01

### Citation

```
@misc{kim2024parahome,
      title={ParaHome: Parameterizing Everyday Home Activities Towards 3D Generative Modeling of Human-Object Interactions},
      author={Jeonghwan Kim and Jisoo Kim and Jeonghyeon Na and Hanbyul Joo},
      year={2024},
      eprint={2401.10232},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Acknowledgement

This work was supported by Samsung Electronics MX Division, NRF grant funded by the Korean government (MSIT) (No. 2022R1A2C2092724 and No. RS-2023-00218601), and IITP grant funded by the Korean government (MSIT) (No.2021-0-01343). H. Joo is the corresponding author.
