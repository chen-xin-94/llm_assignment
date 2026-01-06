You will build a small end-to-end pipeline that collects a manageable amount of recent
BMW-related text from the BMW press releases
(<https://www.press.bmwgroup.com/global/>) and preprocesses it into clean training and
evaluation splits.
Choose a small open-source language model, fine-tune it briefly on this corpus (enough
steps to show the pipeline working), and log basic training information such as loss over
time.
Implement a simple evaluation that computes at least one automatic metric on a held-out
set and produces a few sample generations answering BMW-related prompts.
Deliver a Git repository with the code and a README explaining how to run each step,
your main design choices, and a short summary of the results; the whole task should fit
roughly into 6–8 hours.

Option 1.2: Stretch version
You create a second variant of your chosen language model by removing one transformer
block or hidden layer.
You then fine-tune both the original and the reduced model on the same BMW press-
release corpus under comparable training settings.
You compare their training behaviour (e.g. loss curves), automatic evaluation of metrics
on the held-out set, and a few qualitative BMW-related generations.
Optionally, you design a tiny BMW news Q&A set and report a simple metric such as
accuracy or average log-likelihood for both models.
Finally, in the README you briefly discuss the trade-offs between model size, training
speed, and output quality, and outline what you would investigate next with more time or
compute.
Note: The main goal of the assignment is not to train an optimal model and generate the
highest accuracy but rather proofing a sound understanding of the relevant technical
concepts by selecting sensible decisions and being able to clearly communicate why the
specific strategy was chosen and how it influenced the results. Higher accuracy, larger
models, and similar will not bring you any benefit for the evaluation.
