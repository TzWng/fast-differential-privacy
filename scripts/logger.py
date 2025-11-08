class ExecutionLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path

    def log(self, **kwargs):
        with open(self.path, "a") as f:
            line = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            f.write(line + "\n")
