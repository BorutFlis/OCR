import ocr


ocr_inst = ocr.OcrValidation()

with open("test.json") as f:
    data = f.read()

ocr_inst.new_example_json(data)




