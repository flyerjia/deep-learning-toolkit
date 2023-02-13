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
        return len(self.data)
    def collate_fn(self, batch_data):
        """
        必有
        """
        pass

reader = YourReader // 一定要有
```
```python
class YourModel(BaseModel):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        for name, value in kwargs.items():
            setattr(self, name, value)
        # encoder的名字一定要包含‘encoder’，分层学习率才能生效

    def forward(self, **input_data):
        """
        参数 input_data 要包含phase
        """
        # train 返回loss 其他不返回loss
        raise NotImplementedError

    def get_metrics(self, phase, predictions, dataset):
        """
        计算评价指标, 参数固定

        Args:
            predictions (List): 预测结果
            dataset(Dataset): dataset
        Raises:
            NotImplementedError: 模型单独实现
        """
        raise NotImplementedError

    def get_predictions(self, forward_output, forward_target, dataset, batch_start_index=0):
        """
        计算预测结果，参数固定，对每个batch的数据进行解码

        Args:
            forward_output (Dict): {name:batch_data} batch_data: numpy
            forward_target (Dict): {name:batch_data} batch_data: numpy
            dataset (Dataset): dataset
            batch_start_index (int): 对于dataset中，对应的数据起始位置

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def save_predictions(self, predictions, file_path):
        """
        保存预测结果，参数固定，可根据需求自行设置保存格式
        """
        write_json(file_path, predictions)

model = YourModel // 一定要有
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