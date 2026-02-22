from inference import predict

x = {
    "year": 2019,
    "odometer": 217000,
    "est_value": 41115,
    "make": "GMC",
    "segment": "2500",
    "trim_tier": 3,
}

print(predict(x))
