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
# 若进行部署service_inference，还需要在dltk/service_input下创建对应的参数说明文件
class InputData(BaseModel):
    pid: int
    diagnose: str
    category: str
    ocr_json: List[Dict]
    output_all_keys: bool = False  # 默认值
    name: Union[str, None]  # 可选类型
    debug: Optional[bool] = True  # 可选参数和默认值


input_data = InputData # 一定要有
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
 - multi gpus

注：当使用multi gpus时，实际batch_size = n_gpus * batch_size，应调大学习率进行适配。