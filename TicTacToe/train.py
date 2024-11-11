from ModelFree.DynamicProgramming import DPAgent

model_free = ["DP"]
model_based = []
total_models = model_free + model_based


def init_model(args):
    model_name = args.model_name
    if model_name not in total_models:
        raise ValueError(f"Model {model_name} does not exists. "
                         f"Try one of {total_models}.")

    model = DPAgent(args.size)

    return model


def train(args):
    model = init_model(args)
    model.train()
