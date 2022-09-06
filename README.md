# Learning Linear and Nonlinear Low-Rank Structure in Multi-Task Learning

## Prerequisites

- Python3
- PyTorch ==1.6.0
- Numpy

## Datasets

- fc7 layer feature:
  - office_31_fc7.txt
  - image_clef_fc7.txt
  - office_Home_fc7.txt
  - office-catech_fc7.txt
  - DomainNet_fc7.txt
- pool5 layer feature:
  - office_31_pool5.txt
  - image_clef_pool5.txt
  - office_Home_pool5.txt
  - office-catech_pool5.txt
  - DomainNet_pool5.txt 

Download the pre-processed fc7 layer feature and pool5 layer feature from below link.

[https://drive.google.com/drive/folders/1mAc2MPMIzChruQ6SBUC1eHB3XLSaDXvK?usp=sharing](https://drive.google.com/drive/folders/1mAc2MPMIzChruQ6SBUC1eHB3XLSaDXvK?usp=sharing)

## Training

1. Downloading the dataset(s) from above link.

2. Run the experiment(s) (task fc7 layer as example):

   ```bash
   cd fc7
   python main.py
   ```

   To run this experiment, you need to modify the path `datafile = 'xxx.txt'` before you run `python main.py`.

## Citation

If you use this code for your research, please consider citing:

```
@inproceedings{zhang2021multi,
  title={Multi-task learning via generalized tensor trace norm},
  author={Zhang, Yi and Zhang, Yu and Wang, Wei},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={2254--2262},
  year={2021}
}
@article{zhang2022learning,
  title={Learning Linear and Nonlinear Low-Rank Structure in Multi-Task Learning},
  author={Zhang, Yi and Zhang, Yu and Wang, Wei},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022},
  publisher={IEEE}
}
```

## Contact

If you have any problem about our code, feel free to contact [11930380@mail.sustech.edu.cn](mailto:11930380@mail.sustech.edu.cn).
