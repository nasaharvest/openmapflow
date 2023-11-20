README - NAAMA GLAUBER

* SUMMARY
I improved the model (currently f1=0.72), with these results:
- f1=0.7734, (using pre-trained, I got a few models that improved the f1-score.
Best one is: Togo25epPreTrain_TransOrig-n_ep20-Feb_12-lr001-b_size64-upsm_ratio10.pt (trained on cpu)

- f1=0.7642, (using normalization), Different normalizations improved the f1-score.
Best one is: NormAllCh-n_ep10-Feb_12-lr0001-b_size64-upsm_ratio05.pt (trained on cuda)
---------------------------------------------- 

* STEPS
I was testing several options to improve the model's performance:

1) Running code while changing some parameters:
- Hyper parameters: epochs, batch_size, lr (including torch.optim.lr_scheduler, step, decay), upsampling ratio
- Model's parameters (Transformer): d_model, d_ffn, dropout

----------------------------------------------
2) Normalization the Data
I have seen in the current normalization (using MAX_BANDS), there is no reference to the min values.
However, the min values are necessary (for example I found out that the first two bands are mostly in a range of ~[-40,0], instead of [0,50], as in the current code)...

I checked statistically the values of each band, for 'Togo'(train, val) and 'Global'(train). I didn't used the test for this, as it is not fair...
- I calculated the min, max, std, med, avg: and check normalization with x-min/(max-min) for both global+Togo, and only Togo (the normalization for Togo leads to better results, but I didn't get better with fine-tune over it...)
- I also checked the normalization of 99.9% of max/min to reduce more the noise - not sure if helpful...
- After normalization, since there is noise, I used another pre-processing of one of these: tanh/sigmoid/clipping in order to ensure that the data is within the range [0,1]- wasn't helpful.
- I also added the option to train the model on only several bands by configuration. However, I didn't find a good solution with that (dind't run all the options)
 
----------------------------------------------
3) Fine-Tuning to Togo Area
The purpose of this model is to classify "crop/non-crop" in Togo country (also I see the test and validation sets are only from Togo)...
While normalizing, I have seen that the "Global" and "Togo" have quite different data distribution over the bands (std, mean, as well as min, max values).
Thus, statistically the "Global" data is less representing the test case, and I decided to try this:

3.1) Initially, I added the option of upsampling more data from Togo, and training the model in one step. However, it didn't changed much the results.
code: train_utils/upsample_balance_label_country_df()

3.2) Fine-Tune to Togo countr in two steps of learning:
- First, I trained a model on all of the data (because only Togo-dataset is too small, and more data is necessary).
- Second, I fine-tuned the all-data trained model from the previous step
In both steps, I tried several #eps, lr-s, and with several normalization options. 
code: both in train.py: main(), and load_script_module() to load pre-trained model for farther learning.


----------------------------------------------

* FUTURE
Unfortunatelly, due to time limitation, and running time, I couldn't manage to investigate more than presented above.
However, for future work I think it could be really inteseresting to include the spatial information.
This is not exists in the current presentation of the model- as the data now is 'per-pixel', thus:
- either as augmentation using unsupervised data (if exists data that is really near-by).
- or as another fine-tune for nearest labeled-points (during inference time).

----------------------------------------------

* TECHNICAL INFO
I filtered and cleaned all the non-relevant code, also some code of experiments is in the J.notebook, but if you want that- let me know... 
What was added:
- result-folder: with the logs, metrics, and each with the 'train.py', which was the training code of that model.
- train.py- updated
- since openmapflow is pip-installed it was an issue to edit ./openmapflow/bands.py for the new normalization values (hence they are also in the train.py)...

Thank You!