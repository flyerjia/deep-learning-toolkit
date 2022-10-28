### Deep Learning Toolkit
#### 简介
易用的深度学习训练框架（基于Pytorch）,只需要配置Reader和Model，以及配置文件即可完成模型的training、evaluation、inference、service_inference。

#### 使用说明
```python
class YourReader(BaseReader):
    def __int__(self, phase, data, reader_config):
        super(YourReader, self).__init__(phase, data, config)
        pass
    def convert_item(self, each_data):
        """
        参数不固定，inference阶段必有，训练阶段可无
        """
        pass
    def __getitem__(self, index):
        """
        必有，返回dict格式结果
        """
        pass
    def __len__(self):
        """
        必有
        """
        pass
    def collate_fn(self, batch_data):
        """
        必有
        """
        pass
```
```python
class YourModel(BaseModel):
    def __int__(self, **kwargs):
        super(YourModel, self).__init__(**kwargs)
        # encoder的名字一定要包含‘encoder’，分层学习率才能生效
        pass
    def forward(self, **input_data):
        # input_data为dataset返回dict格式结果 + {'phase': 'train'/'def'/'test'/'inference'}

        pass
    def get_metrics(self, phase, forward_output, forward_target, dataset=None):
        """
        计算评价指标, 参数固定

        Args:
            phase (str): 'training' 'eval' 'inference'等
            forward_output (Dict): {name:[batch1, batch2,...]} batch: numpy
            forward_target (Dict): {name:[batch1, batch2,...]} batch: numpy
            dataset(Dataset): dataset
        Return:
            Dict: 包含各种指标的dict
        """
        pass
    def get_predictions(self, forward_output, dataset):
        """
        计算预测结果，参数固定

        Args:
            forward_output (Dict): {name:[batch1, batch2,...]} batch: numpy
            dataset (Dataset): dataset
        Return:
            List[Dict]
        """
        pass
    def save_predictions(self, forward_output, dataset, file_path):
        """
        保存预测结果，参数固定
        """
        pass
```

```python
# 若进行部署service_inference，还需要修改BaseController.service()函数的InputData，进行网络接口参数的定制化操作
def service(self):
    dataset = self.init_dataset_inference()
    inference = self.init_inference()

    app = FastAPI()

    class InputData(BaseModel):
        # 自行定义
        text: str

    @app.get("/health.json")
    def health():
        return {"status": "UP"}

    # 接口均可自行定义
    @app.post('/innerapi/ai/python/agent')
    async def default_interface(input_data: InputData):
        try:
            result = inference.service_inference(dataset, input_data.dict())
        except Exception as ex:
            return {
                'code': -1,
                'msg': ex
            }
        return {
            'code': 0,
            'msg': 'OK',
            'data': result
        }
    uvicorn.run(app, host='0.0.0.0', port=8080)
```

```shell
python -m dltk --command training/training_cv/inference/service --config my_config.yaml
```

#### 已支持操作
 - training
 - training_cv
 - evaluation
 - inference
 - service_inference
 - early_stopping
 - FGM
 - EMA
 - FP16
 - convert pt model to tf model

