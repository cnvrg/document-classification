from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer, sample_dataset
import os
from sklearn.metrics import confusion_matrix

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

# Load a dataset from the Hugging Face Hub
def run_training(data, batch_size, num_iterations, num_epochs, model_id, le):

    # Load a SetFit model from Hub
    model = SetFitModel.from_pretrained(model_id)
    print("model has been loaded")
    # Create trainer

    if len(list(data.keys())) == 2:

        trainer = SetFitTrainer(
            model=model,
            train_dataset=data["train"],
            loss_class=CosineSimilarityLoss,
            eval_dataset=data["eval"],
            metric="accuracy",
            batch_size=batch_size,
            num_iterations=num_iterations,  # The number of text pairs to generate for contrastive learning
            num_epochs=num_epochs,  # The number of epochs to use for contrastive learning
        )
        trainer.train()
        metrics = trainer.evaluate()

        print("Accuracy of the trained model is: ", metrics)
        pred = trainer.model.predict(data["eval"]["text"])
        true = data["eval"]["label"]
        cm = confusion_matrix(true, pred)

        import seaborn as sns
        import matplotlib.pyplot as plt

        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt="g", ax=ax)

        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")
        ax.xaxis.set_ticklabels(le.classes_)
        ax.yaxis.set_ticklabels(le.classes_)
        image = ax.figure
        image.savefig(cnvrg_workdir+"/confusion_matrix.png")

    else:

        trainer = SetFitTrainer(
            model=model,
            train_dataset=data["train"],
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
            batch_size=batch_size,
            num_iterations=num_iterations,  # The number of text pairs to generate for contrastive learning
            num_epochs=num_epochs,  # The number of epochs to use for contrastive learning
        )
        print("trainer has been created and starting training")
        trainer.train()

    trainer.model._save_pretrained(save_directory=cnvrg_workdir)
