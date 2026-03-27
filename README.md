

```shell
python setup.py install
```


```shell
# 单测
python test_scaled_mm.py 
```




```py
import cutlass_scaled_mm_paddle
result = cutlass_scaled_mm_paddle.cutlass_scaled_mm(a, b, scale_a, scale_b, bias)
```
