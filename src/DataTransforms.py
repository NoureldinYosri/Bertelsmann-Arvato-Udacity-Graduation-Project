from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from tqdm import tqdm

    
class baseTransform:
    def __init__(self):
        pass
    def apply(self, col):
        raise NotImplementedError

class Drop(baseTransform):
    def __init__(self):
        super().__init__()
    
    def apply(self, col):
        return None
    
class Replace(baseTransform):
    def __init__(self, old_value, new_value):
        super().__init__()
        self.old_value = old_value
        self.new_value = new_value
        
    def apply(self, col):
        if self.old_value is None:
            return col.fillna(self.new_value)
        
        return col.replace(self.old_value, self.new_value)

class OneHotTransform(baseTransform):
    def __init__(self, col):
        super().__init__()
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder.fit(col.values.reshape(-1,1))
        
    
    def apply(self, col):
        cname = col.name
        df = pd.DataFrame(self.encoder.transform(col.values.reshape(-1,1)).todense())
        df.columns = [cname+'$'+str(i) for i in df.columns]
        return df

    
class CustomTransform(baseTransform):
    def __init__(self, f):
        self.f = f
    def apply(self, col):
        return col.apply(self.f)
    
    
def TransformData(inputData, transforms):
    clean_data = []
    for col in tqdm(transforms):
        T = transforms[col]
        data = inputData[col]
        for t in T:        
            if isinstance(t, str):
                assert (t == 'cast')
                data = data.astype('float64')
            else:
                try:
                    data = t.apply(data)
                except:
                    data = pd.DataFrame(t.transform(data.values.reshape(-1, 1)))
                    data.columns = [col]
        if data is None: continue
        clean_data.append(data)
    data = None
    return pd.concat(clean_data, axis = 1)
