from huggingface_hub import HfApi, ModelFilter

api = HfApi()
models = api.list_models(filter=ModelFilter(task="")) #specify the task name
for model in models:
    print(model.modelId)
