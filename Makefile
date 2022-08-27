build:
	rm -rf ../dist
	rm -rf ../build
	pyinstaller -D -i test.ico main.py
	mv dist ..
	mv build ..

zip:
	zip


clean:
	pyinstaller --clean main.py


run:
	python main.py


upload:
	curl -X DELETE http://sunie.tpddns.cn:9052/files/digital-image.zip
	curl -F file=@../digital-image.zip http://sunie.tpddns.cn:9052/files
