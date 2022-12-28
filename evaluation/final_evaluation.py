# from helpers.simple_models import evaluate_simple_models
# from datasets import load_dataset
# from evaluate_models import *
#
#
# def predict_with_lstm(train_dataset, test_dataset):
#     metaphor_bin_dataset = load_dataset("../datasets/MetaphorBinDataset.py")
#     predictions = metaphor_bin_dataset['test']['labels']
#     return predictions
#
#
# def final_evaluation(models_list, models_to_show):
#     """
#     This function is used to evaluate the final models. It is used to evaluate the final models on the test set.
#     :param models_list: list of models to evaluate
#     :param models_to_show: number of models to show
#     """
#
#     # Load dataset
#     metaphor_bin_dataset = load_dataset("../datasets/MetaphorBinDataset.py")
#     test_dataset = metaphor_bin_dataset['test']
#     train_dataset = metaphor_bin_dataset['train']
#
#
#     test_dataset_name = "YBYDataset"
#     # Calculate metrics for simple models
#     eval_test_majority, eval_test_constant_false = evaluate_simple_models(train_dataset, test_dataset,
#                                                                           test_dataset_name)
#     words_statistics, metaphor_words, non_metaphor_words = \
#         calculate_words_statistics(train_dataset, dataset_name="MetaphorBinDataset")
#
#     # evaluate trained lstm model
#     lstm_predictions = predict_with_lstm(train_dataset, test_dataset)
#     # # save test_dataset['data'] and test_dataset['labels'] lists to pickle files
#     # with open('data.pkl', 'wb') as f:
#     #     pickle.dump(train_dataset['data'], f)
#     # with open('labels.pkl', 'wb') as f:
#     #     pickle.dump(train_dataset['labels'], f)
#
#     lstm_eval = full_eval(data=test_dataset['data'], predictions=lstm_predictions, labels=test_dataset['labels'],
#                           metaphor_words=metaphor_words, non_metaphor_words=non_metaphor_words)
#
#     eval_results_other_models = {'majority': eval_test_majority, 'constant_false': eval_test_constant_false,
#                                  'lstm': lstm_eval}
#
#     # initialize dictionary to store metrics
#     metrics_dict = {'total': {'f1_scores': {}, 'accuracy_scores': {}, 'precision_scores': {}, 'recall_scores': {}},
#                     'seen': {'f1_scores': {}, 'accuracy_scores': {}, 'precision_scores': {}, 'recall_scores': {}},
#                     'unseen': {'f1_scores': {}, 'accuracy_scores': {}, 'precision_scores': {}, 'recall_scores': {}}}
#
#     for model, eval_result in eval_results_other_models.items():
#         metrics_dict['total']['f1_scores'][model] = eval_result['total']['f1']
#         metrics_dict['total']['accuracy_scores'][model] = eval_result['total']['accuracy']
#         metrics_dict['total']['precision_scores'][model] = eval_result['total']['precision']
#         metrics_dict['total']['recall_scores'][model] = eval_result['total']['recall']
#         metrics_dict['seen']['f1_scores'][model] = eval_result['seen']['f1']
#         metrics_dict['seen']['accuracy_scores'][model] = eval_result['seen']['accuracy']
#         metrics_dict['seen']['precision_scores'][model] = eval_result['seen']['precision']
#         metrics_dict['seen']['recall_scores'][model] = eval_result['seen']['recall']
#         metrics_dict['unseen']['f1_scores'][model] = eval_result['unseen']['f1']
#         metrics_dict['unseen']['accuracy_scores'][model] = eval_result['unseen']['accuracy']
#         metrics_dict['unseen']['precision_scores'][model] = eval_result['unseen']['precision']
#         metrics_dict['unseen']['recall_scores'][model] = eval_result['unseen']['recall']
#
#     # initialize dictionary to store metrics
#     f1_scores_trained, accuracy_scores_trained, recall_scores_trained, precision_scores_trained = {}, {}, {}, {}
#     eval_trained_bert_dict = {}
#
#     # evaluate trained bert models
#     for checkpoint in models_list:
#         checkpoint = checkpoint.replace('results/checkpoints/', '')
#         eval_test_trained = evaluate_model(checkpoint, train_dataset, test_dataset, test_dataset_name)
#         eval_trained_bert_dict[checkpoint] = eval_test_trained
#
#     # iterate over keys and values of the eval_trained_dict
#     for checkpoint, eval_result in eval_trained_bert_dict.items():
#         f1_scores_trained[checkpoint] = eval_result['total']['f1']
#         accuracy_scores_trained[checkpoint] = eval_result['total']['accuracy']
#         recall_scores_trained[checkpoint] = eval_result['total']['recall']
#         precision_scores_trained[checkpoint] = eval_result['total']['precision']
#
#     # keep only the number_of_models best models results (highest f1 score) and change the names
#     f1_scores_trained_sorted = sorted(f1_scores_trained.items(), key=lambda x: x[1], reverse=True)[:models_to_show]
#
#     best_models_names = {}
#     name_list = ['majority', 'constant_false', 'lstm']
#     for i, (name, f1_score) in enumerate(f1_scores_trained_sorted):
#         model_name = "model {}".format(i + 1)
#         name_list.append(model_name)
#         metrics_dict['total']['f1_scores'][model_name] = f1_scores_trained[name]
#         metrics_dict['total']['accuracy_scores'][model_name] = accuracy_scores_trained[name]
#         metrics_dict['total']['precision_scores'][model_name] = precision_scores_trained[name]
#         metrics_dict['total']['recall_scores'][model_name] = recall_scores_trained[name]
#         metrics_dict['seen']['f1_scores'][model_name] = eval_trained_bert_dict[name]['seen']['f1']
#         metrics_dict['seen']['accuracy_scores'][model_name] = eval_trained_bert_dict[name]['seen']['accuracy']
#         metrics_dict['seen']['precision_scores'][model_name] = eval_trained_bert_dict[name]['seen']['precision']
#         metrics_dict['seen']['recall_scores'][model_name] = eval_trained_bert_dict[name]['seen']['recall']
#         metrics_dict['unseen']['f1_scores'][model_name] = eval_trained_bert_dict[name]['unseen']['f1']
#         metrics_dict['unseen']['accuracy_scores'][model_name] = eval_trained_bert_dict[name]['unseen']['accuracy']
#         metrics_dict['unseen']['precision_scores'][model_name] = eval_trained_bert_dict[name]['unseen']['precision']
#         metrics_dict['unseen']['recall_scores'][model_name] = eval_trained_bert_dict[name]['unseen']['recall']
#         best_models_names[name] = name_list
#
#     for type in metrics_dict.keys():
#         plt.figure(figsize=(15, 15))
#         for score_index, score in enumerate(metrics_dict[type].keys()):
#             plt.subplot(2, 2, score_index + 1)
#             plt.title("{}".format(score))
#             # Next two models are the simple models, color them differently
#             plt.bar(name_list[:2], [metrics_dict['total'][score][name_list[0]],
#                                     metrics_dict['total'][score][name_list[1]]], color='#1f77b4')
#             # Next model is lstm, color it differently
#             plt.bar(name_list[2], metrics_dict['total'][score][name_list[2]], color='#ff7f0e')
#             for index in range(3, len(name_list)):
#                 plt.bar(name_list[index], metrics_dict['total'][score][name_list[index]], color='#2ca02c')
#             # make x-axis ticks readable
#             plt.xticks(rotation=90)
#             # show the scores on the bar chart up to 2 decimal places
#             for i, v in enumerate(metrics_dict['total'][score].values()):
#                 plt.text(i - 0.15, v + 0.01, '{:.2f}'.format(v), color='black', fontweight='bold')
#             plt.tight_layout()
#         plt.tight_layout()
#         # save the plot with date
#         path_to_plots = 'results/plots'
#         title = '{} scores_trained_models ({})'.format(test_dataset_name, type)
#         if not os.path.exists(path_to_plots):
#             os.makedirs(path_to_plots)
#         plt.savefig("{}/{}_{}_{}.png".format(path_to_plots, title, type, str(datetime.now())))
#         # save f1_scores_sorted at txt file to the same folder
#         with open("{}/{}_{}_{}.txt".format(path_to_plots, title, type, str(datetime.now())), "w") as text_file:
#             text_file.write(str(f1_scores_trained_sorted))
#         plt.show()
#
#     # plot confusion matrix
#     for checkpoint, eval_results in eval_trained_bert_dict.items():
#         for type in eval_results.keys():
#             # remove '/' from the name of the model
#             checkpoint = checkpoint.replace('/', '_')
#             title = 'Confusion matrix of model {} on {} ({})'.format(checkpoint, test_dataset_name, type)
#             create_confusion_matrix(eval_results[type]['cf_matrix'], title=title,
#                                     filename='results/plots/{}_{}'.format(checkpoint, type))
#     for checkpoint, eval_results in eval_results_other_models.items():
#         for type in eval_results.keys():
#             title = 'Confusion matrix of model {} on {} ({})'.format(checkpoint, test_dataset_name, type)
#             create_confusion_matrix(eval_results[type]['cf_matrix'], title=title,
#                                     filename='results/plots/{}_{}'.format(checkpoint, type))
#
#     print("Done")
#
#
# if __name__ == '__main__':
#     models_to_show = 5
#     checkpoint_list = [x[0] for x in os.walk('results/checkpoints')][1:]
#     checkpoint_after_pre = '/home/tok/figurative-language/onlplab/alephbert-base_fbt_32epochs_trainer/'
#     checkpoint_list.append(checkpoint_after_pre)
#     final_evaluation(models_list=checkpoint_list, models_to_show=models_to_show)
#
