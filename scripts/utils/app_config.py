# Define a class to hold the configuration
class AppConfig:
    def __init__(self, data_file, output_file, rest_endpoint, model_name, max_tokens, max_new_tokens, temperature, max_words):
        self.data_file = data_file
        self.output_file = output_file
        self.rest_endpoint = rest_endpoint
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_words =  max_words

    def __repr__(self):
        return (f"Config(data_file={self.data_file}, output_file={self.output_file}, "
                f"rest_endpoint={self.rest_endpoint}, model_name={self.model_name}, "
                f"max_tokens={self.max_tokens}, max_new_tokens={self.max_new_tokens}, "
                f"temperature={self.temperature}, max_words= {self.max_words})")