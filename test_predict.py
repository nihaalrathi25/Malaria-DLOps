from src.inference.predict import predict_image

result = predict_image("test_image.png")  # put any malaria cell image here
print("Prediction:", result)