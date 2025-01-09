Commands:  
```python GMF_torch.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1```  

```python MLP_torch.py --dataset ml-1m --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1```  

```python NeuMF_torch.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1```  

```python NeuMF_torch.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1 --mf_pretrain BestModel/best_gmf_model.pth --mlp_pretrain BestModel/best_mlp_model.pth```  

```python NeuMF_torch.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1```

Dropout Regularization - Generalize model preventing overfitting (Reducing the weight)
Layer Normalization - For stability, prevent underflow and may improve performance by using low numbers (data rescaling)
Learning Rate Scheduling - Dynamically reduce learning rate

Metrics:
- Hit Rate (HR) measures the proportion of times the true item appears in the top-K recommendations.
- Loss (binary cross-entropy) measures the difference between the predicted probabilities and the actual binary labels.
- Normalized Discounted Cumulative Gain (NDCG) evaluates the ranking quality of the recommended items, giving higher scores to correct recommendations that appear higher in the list.

GMF HR: 0.0025, Best NDCG: 0.0011
MLP HR: 0.0227, Best NDCG: 0.0094
NeuMF (standard) HR: 0.0235, Best NDCG: 0.0116
NeuMF (pretrained) HR: 0.0240, Best NDCG: 0.0108
NeuMF (adjusted) HR: 0.0262, Best NDCG: 0.0123

Best Model: NeuMF (pretrained)
Best Ranking: NeuMF (standard)

# GPT Analysis
GMF:
- The GMF model has the lowest HR and NDCG among all models.
- This indicates that the GMF model is less effective in making accurate recommendations and ranking them correctly.
MLP:
- The MLP model shows a significant improvement in HR and NDCG compared to the GMF model.
- This suggests that the MLP model is better at capturing the interactions between users and items through its multi-layer architecture.
NeuMF (standard):
- The standard NeuMF model further improves HR and NDCG compared to the MLP model.
- This indicates that combining the strengths of both GMF and MLP models leads to better performance.
NeuMF (pretrained):
- The pretrained NeuMF model achieves the highest HR (0.0240) and a slightly lower NDCG (0.0108) compared to the standard NeuMF model.
- The slight decrease in NDCG might be due to overfitting or other factors, but the overall performance is still strong.

# GMF
Epoch 1/20, Loss: 1.2999622124736592
Epoch 1: HR = 0.0022, NDCG = 0.0010
Epoch 2/20, Loss: 1.317446334887359
Epoch 2: HR = 0.0023, NDCG = 0.0011
Epoch 3/20, Loss: 1.31208833698499
Epoch 3: HR = 0.0023, NDCG = 0.0011
Epoch 4/20, Loss: 1.294463589029797
Epoch 4: HR = 0.0022, NDCG = 0.0010
Epoch 5/20, Loss: 1.2756653276540466
Epoch 5: HR = 0.0022, NDCG = 0.0010
Epoch 6/20, Loss: 1.2779145018529083
Epoch 6: HR = 0.0022, NDCG = 0.0010
Epoch 7/20, Loss: 1.2596750804933452
Epoch 7: HR = 0.0022, NDCG = 0.0010
Epoch 8/20, Loss: 1.2510839993670835
Epoch 8: HR = 0.0022, NDCG = 0.0010
Epoch 9/20, Loss: 1.2477247189667264
Epoch 9: HR = 0.0020, NDCG = 0.0010
Epoch 10/20, Loss: 1.2411915619494551
Epoch 10: HR = 0.0022, NDCG = 0.0010
Epoch 11/20, Loss: 1.2504368939642179
Epoch 11: HR = 0.0025, NDCG = 0.0011
Epoch 12/20, Loss: 1.2212548937837957
Epoch 12: HR = 0.0020, NDCG = 0.0010
Epoch 13/20, Loss: 1.2154580228409524
Epoch 13: HR = 0.0022, NDCG = 0.0010
Epoch 14/20, Loss: 1.20824578454939
Epoch 14: HR = 0.0022, NDCG = 0.0010
Epoch 15/20, Loss: 1.2008229630478358
Epoch 15: HR = 0.0020, NDCG = 0.0010
Epoch 16/20, Loss: 1.1882136292376761
Epoch 16: HR = 0.0020, NDCG = 0.0010
Epoch 17/20, Loss: 1.191640606876147
Epoch 17: HR = 0.0022, NDCG = 0.0010
Epoch 18/20, Loss: 1.190582522901438
Epoch 18: HR = 0.0020, NDCG = 0.0010
Epoch 19/20, Loss: 1.164238426139799
Epoch 19: HR = 0.0022, NDCG = 0.0010
Epoch 20/20, Loss: 1.1574499101962072
Epoch 20: HR = 0.0022, NDCG = 0.0010
Best HR: 0.0025, Best NDCG: 0.0011

# MLP
Epoch 1/20, Loss: 0.5977465409343525
Epoch 1: HR = 0.0050, NDCG = 0.0019
Epoch 2/20, Loss: 0.49724997832613477
Epoch 2: HR = 0.0056, NDCG = 0.0024
Epoch 3/20, Loss: 0.487791755694454
Epoch 3: HR = 0.0078, NDCG = 0.0040
Epoch 4/20, Loss: 0.47616223384768275
Epoch 4: HR = 0.0114, NDCG = 0.0052
Epoch 5/20, Loss: 0.4628950423103268
Epoch 5: HR = 0.0123, NDCG = 0.0055
Epoch 6/20, Loss: 0.44648133243544624
Epoch 6: HR = 0.0124, NDCG = 0.0059
Epoch 7/20, Loss: 0.432255003159329
Epoch 7: HR = 0.0166, NDCG = 0.0071
Epoch 8/20, Loss: 0.41699564785270365
Epoch 8: HR = 0.0129, NDCG = 0.0060
Epoch 9/20, Loss: 0.4027964184849949
Epoch 9: HR = 0.0149, NDCG = 0.0075
Epoch 10/20, Loss: 0.3930153000657841
Epoch 10: HR = 0.0184, NDCG = 0.0085
Epoch 11/20, Loss: 0.38795374157065055
Epoch 11: HR = 0.0187, NDCG = 0.0084
Epoch 12/20, Loss: 0.38091844067735187
Epoch 12: HR = 0.0187, NDCG = 0.0084
Epoch 13/20, Loss: 0.37733330539727616
Epoch 13: HR = 0.0184, NDCG = 0.0081
Epoch 14/20, Loss: 0.3692509246579671
Epoch 14: HR = 0.0199, NDCG = 0.0086
Epoch 15/20, Loss: 0.3706582890728773
Epoch 15: HR = 0.0205, NDCG = 0.0086
Epoch 16/20, Loss: 0.3678592655618312
Epoch 16: HR = 0.0227, NDCG = 0.0094
Epoch 17/20, Loss: 0.36523921055308844
Epoch 17: HR = 0.0204, NDCG = 0.0088
Epoch 18/20, Loss: 0.3646698707746247
Epoch 18: HR = 0.0205, NDCG = 0.0091
Epoch 19/20, Loss: 0.3574926618297221
Epoch 19: HR = 0.0212, NDCG = 0.0088
Epoch 20/20, Loss: 0.3618281716007297
Epoch 20: HR = 0.0207, NDCG = 0.0087
Best HR: 0.0227, Best NDCG: 0.0094

# NeuMF (standard)
Epoch 1/20, Loss: 0.6049457159587892
Epoch 1: HR = 0.0030, NDCG = 0.0013
Epoch 2/20, Loss: 0.499665836914111
Epoch 2: HR = 0.0046, NDCG = 0.0022
Epoch 3/20, Loss: 0.48805627403622964
Epoch 3: HR = 0.0070, NDCG = 0.0032
Epoch 4/20, Loss: 0.4806237612235344
Epoch 4: HR = 0.0101, NDCG = 0.0051
Epoch 5/20, Loss: 0.4682508002398378
Epoch 5: HR = 0.0127, NDCG = 0.0058
Epoch 6/20, Loss: 0.4538908191656662
Epoch 6: HR = 0.0144, NDCG = 0.0061
Epoch 7/20, Loss: 0.4432606601108939
Epoch 7: HR = 0.0161, NDCG = 0.0074
Epoch 8/20, Loss: 0.4275890550876068
Epoch 8: HR = 0.0171, NDCG = 0.0081
Epoch 9/20, Loss: 0.4136034838728986
Epoch 9: HR = 0.0162, NDCG = 0.0078
Epoch 10/20, Loss: 0.39803787257711765
Epoch 10: HR = 0.0204, NDCG = 0.0094
Epoch 11/20, Loss: 0.39186718706357276
Epoch 11: HR = 0.0199, NDCG = 0.0091
Epoch 12/20, Loss: 0.3825362284304732
Epoch 12: HR = 0.0222, NDCG = 0.0105
Epoch 13/20, Loss: 0.37631995647640554
Epoch 13: HR = 0.0215, NDCG = 0.0109
Epoch 14/20, Loss: 0.37521937085410295
Epoch 14: HR = 0.0235, NDCG = 0.0116
Epoch 15/20, Loss: 0.371164820962033
Epoch 15: HR = 0.0220, NDCG = 0.0110
Epoch 16/20, Loss: 0.36625732430967234
Epoch 16: HR = 0.0227, NDCG = 0.0115
Epoch 17/20, Loss: 0.36407876646114606
Epoch 17: HR = 0.0222, NDCG = 0.0109
Epoch 18/20, Loss: 0.3623258744255971
Epoch 18: HR = 0.0220, NDCG = 0.0107
Epoch 19/20, Loss: 0.3645785718651141
Epoch 19: HR = 0.0227, NDCG = 0.0111
Epoch 20/20, Loss: 0.36449359887737337
Epoch 20: HR = 0.0230, NDCG = 0.0116
Best HR: 0.0235, Best NDCG: 0.0116

# NeuMF (pretrained)
Loaded pretrained GMF model from BestModel/best_gmf_model.pth and MLP model from BestModel/best_mlp_model.pth
Epoch 1/20, Loss: 0.3953452145649215
Epoch 1: HR = 0.0171, NDCG = 0.0072
Epoch 2/20, Loss: 0.36998856269707114
Epoch 2: HR = 0.0172, NDCG = 0.0073
Epoch 3/20, Loss: 0.36894383521403296
Epoch 3: HR = 0.0182, NDCG = 0.0078
Epoch 4/20, Loss: 0.3621592981330419
Epoch 4: HR = 0.0182, NDCG = 0.0079
Epoch 5/20, Loss: 0.35841880587197966
Epoch 5: HR = 0.0220, NDCG = 0.0090
Epoch 6/20, Loss: 0.35835496045775334
Epoch 6: HR = 0.0197, NDCG = 0.0085
Epoch 7/20, Loss: 0.35934601193767485
Epoch 7: HR = 0.0209, NDCG = 0.0088
Epoch 8/20, Loss: 0.3551764851909573
Epoch 8: HR = 0.0217, NDCG = 0.0088
Epoch 9/20, Loss: 0.35858527154235514
Epoch 9: HR = 0.0199, NDCG = 0.0083
Epoch 10/20, Loss: 0.3542400095927513
Epoch 10: HR = 0.0204, NDCG = 0.0090
Epoch 11/20, Loss: 0.3568783612069437
Epoch 11: HR = 0.0210, NDCG = 0.0092
Epoch 12/20, Loss: 0.35411885961637657
Epoch 12: HR = 0.0210, NDCG = 0.0091
Epoch 13/20, Loss: 0.35284872353076935
Epoch 13: HR = 0.0202, NDCG = 0.0093
Epoch 14/20, Loss: 0.35627689957618713
Epoch 14: HR = 0.0205, NDCG = 0.0092
Epoch 15/20, Loss: 0.35733484767251095
Epoch 15: HR = 0.0210, NDCG = 0.0091
Epoch 16/20, Loss: 0.35448995762962404
Epoch 16: HR = 0.0233, NDCG = 0.0104
Epoch 17/20, Loss: 0.35680252338870094
Epoch 17: HR = 0.0240, NDCG = 0.0108
Epoch 18/20, Loss: 0.3522720013634633
Epoch 18: HR = 0.0237, NDCG = 0.0103
Epoch 19/20, Loss: 0.35003382371643843
Epoch 19: HR = 0.0237, NDCG = 0.0108
Epoch 20/20, Loss: 0.35379826820502847
Epoch 20: HR = 0.0230, NDCG = 0.0102
Best HR: 0.0240, Best NDCG: 0.0108

# NewMF (adjusted)
Epoch 1/20, Loss: 0.6437993281978672
Epoch 1: HR = 0.0036, NDCG = 0.0017
Epoch 2/20, Loss: 0.502696628035125
Epoch 2: HR = 0.0058, NDCG = 0.0025
Epoch 3/20, Loss: 0.48945841496273623
Epoch 3: HR = 0.0094, NDCG = 0.0042
Epoch 4/20, Loss: 0.4765724885766789
Epoch 4: HR = 0.0108, NDCG = 0.0046
Epoch 5/20, Loss: 0.4646037248736721
Epoch 5: HR = 0.0118, NDCG = 0.0052
Epoch 6/20, Loss: 0.44297836468381396
Epoch 6: HR = 0.0126, NDCG = 0.0054
Epoch 7/20, Loss: 0.42920561538914503
Epoch 7: HR = 0.0124, NDCG = 0.0069
Epoch 8/20, Loss: 0.41462573838435995
Epoch 8: HR = 0.0171, NDCG = 0.0088
Epoch 9/20, Loss: 0.3952200304148561
Epoch 9: HR = 0.0214, NDCG = 0.0104
Epoch 10/20, Loss: 0.393292467725479
Epoch 10: HR = 0.0237, NDCG = 0.0113
Epoch 11/20, Loss: 0.3865415236707461
Epoch 11: HR = 0.0242, NDCG = 0.0116
Epoch 12/20, Loss: 0.375479393338753
Epoch 12: HR = 0.0242, NDCG = 0.0117
Epoch 13/20, Loss: 0.37886641465001186
Epoch 13: HR = 0.0262, NDCG = 0.0123
Epoch 14/20, Loss: 0.3741841803667909
Epoch 14: HR = 0.0238, NDCG = 0.0119
Epoch 15/20, Loss: 0.3702968386775356
Epoch 15: HR = 0.0240, NDCG = 0.0119
Epoch 16/20, Loss: 0.36337882937011073
Epoch 16: HR = 0.0243, NDCG = 0.0116
Epoch 17/20, Loss: 0.36412986279544185
Epoch 17: HR = 0.0233, NDCG = 0.0111
Epoch 18/20, Loss: 0.36165532772823916
Epoch 18: HR = 0.0238, NDCG = 0.0113
Epoch 19/20, Loss: 0.36093722132302947
Epoch 19: HR = 0.0230, NDCG = 0.0110
Epoch 20/20, Loss: 0.3595528925879527
Epoch 20: HR = 0.0235, NDCG = 0.0114
Best HR: 0.0262, Best NDCG: 0.0123