This repo is just some personal investigations into possible exploitable patterns within specific card shuffles.
The repo is currently very poorly organised and the code is a mess, but if you look at the jupyter notebook you might get some semblance of an idea of what I'm doing.

Currently I'm able to predict each next card in a shuffled deck of cards at over twice the accuracy of just guessing.
This method uses a different model for each card position with the data the model takes in being the cards that have been dealt up to that point.

![ensemble_accuracies](https://github.com/user-attachments/assets/0e4c9060-a0f0-41f7-84ec-09ca54b30051)

The models used at each position were found using a grid search over several different machine learning model types. 
The models are a combination of fully connected networks, several forms of recurrent neural networks and transformers.

![performance_plot](https://github.com/user-attachments/assets/34a3e3f1-04b7-4bb0-8110-edc4bb0e6a19)

The transformer models were most accurate in the later stages of the deck with the rnns being accurate at the start and middle and fcnns being most accurate right at the start.

The models were trained on a generated dataset of 10,000 shuffled decks and evaluated on a further 2000.
The shuffle chosen for the test was riffle - box - riffle - box - riffle - box, with randomness parameters chosen to try and mimic real life but I just did this by eye.
A greater number of shuffles in the dataset increases the accuracy well but I wanted to choose a number of shuffles that would be perhaps possible to obtain in the real world.

I will be doing further work on this to refine a card prediciton model whilst also hopefully evaluating its performance and utility in real world games such as blackjack and baccarat.
I also want to see how the predicition performs when the shoe is larger than one deck and therefore duplicates of cards will be present within the deck.
I expect it wouldn't have an effect beyond requiring more data for the statistical patterns to become apparent but this remains to be tested.

Work has also been done on the classification of shuffle method used on a shuffled deck of cards - using just unsipervised methods, cards can be separated, with reasonable accuracy, by how they were shuffled.

![shuffle_analysis5](https://github.com/user-attachments/assets/7a6e229b-b375-4d64-9702-8036b26d03d4)

I lost the actual accuracy for this and the jupyter notebook is now no longer working properly so you'll just have to take my word for it.

Supervised classification of shuffle method can also be performed on these decks with a random forest model getting 85% accuracy on a dataset with a random shuffle and the shuffle used in the predicition experiment.
This is quite interesting as it further shows that cards shuffed in a certain way have identifiable patterns and these can be used to identify them and possibly be used to gain a prediciton advantage.
