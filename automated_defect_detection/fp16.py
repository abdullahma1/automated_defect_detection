from onnxconverter_common import float16
import onnx

input_model = "trained/classifier/best.onnx"
output_model = "trained/classifier/best_fp16.onnx"

print("Loading model...")
model = onnx.load(input_model)

print("Converting to FP16...")
model_fp16 = float16.convert_float_to_float16(model)

print("Saving...")
onnx.save(model_fp16, output_model)

print("DONE! Saved as:", output_model)
