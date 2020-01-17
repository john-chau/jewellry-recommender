from flask import Flask, render_template, request
from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
 
# Create the application object
app = Flask(__name__)

#loading model
learn = load_learner('./', 'img_reco_80_main')
model = learn.model

#loading mejuri product info
mejuri_product_df = pd.read_csv('feat_vectors_final.csv')
mejuri_product_df['img_repr'] = mejuri_product_df['img_repr'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

#hook class and fucntions
class Hook():
    def __init__(self, m:nn.Module, hook_func:HookFunc, is_forward:bool=True, detach:bool=True):
        self.hook_func,self.detach,self.stored = hook_func,detach,None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module:nn.Module, input:Tensors, output:Tensors):
        if self.detach:
            input  = (o.detach() for o in input ) if is_listy(input ) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        if not self.removed:
            self.hook.remove()
            self.removed=True

    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()
        
def get_output(module, input_value, output):
    return output.flatten(1)

def get_input(module, input_value, output):
    return list(input_value)[0]

def get_named_module_from_model(model, name):
    for n, m in model.named_modules():
        if n == name:
            return m
    return None

@app.route('/', methods=('GET', 'POST'))
def home_page():
    # if server request pull, render form
    if request.method == 'GET':
        return render_template("home.html")  # render a template
    
     # if server request push, do this:
    if request.method == 'POST':
       temp_mejuri_product_df = mejuri_product_df
       
       file = request.files['file']
       x = open_image(file)
       
       xb, _ = learn.data.one_item(x)
       
       device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
       xb = xb.to(device)
       
       linear_output_layer = get_named_module_from_model(model, '1.4')
       
       with Hook(linear_output_layer, get_output, True, True) as hook:
          bs = xb.shape[0]
          result = model.eval()(xb)
          base_vector = hook.stored.cpu().numpy()
          base_vector = base_vector.reshape(bs, -1)

          
       cosine_similarity = 1 - temp_mejuri_product_df['img_repr'].apply(lambda x: cosine(x, base_vector))
       similar_img_ids = np.argsort(cosine_similarity)[-3:][::-1]
       
       meijuri_product_reco = temp_mejuri_product_df.iloc[similar_img_ids].reset_index()
       
#       item_list = [0, 1, 2]
#       description_list = ['main_img', 'label', 'product_name', 'material', 'price', 'product_link']
       
       similar1 = meijuri_product_reco.loc[0, 'main_img']
       label1 = meijuri_product_reco.loc[0, 'label']
       product1 = meijuri_product_reco.loc[0, 'product_name']
       material1 = meijuri_product_reco.loc[0, 'material']
       price1 = meijuri_product_reco.loc[0, 'price_CAD']
       product_link1 = meijuri_product_reco.loc[0, 'product_link']
       
       similar2 = meijuri_product_reco.loc[1, 'main_img']
       label2 = meijuri_product_reco.loc[1, 'label']
       product2 = meijuri_product_reco.loc[1, 'product_name']
       material2 = meijuri_product_reco.loc[1, 'material']
       price2 = meijuri_product_reco.loc[1, 'price_CAD']
       product_link2 = meijuri_product_reco.loc[1, 'product_link']
       
       similar3 = meijuri_product_reco.loc[2, 'main_img']
       label3 = meijuri_product_reco.loc[2, 'label']
       product3 = meijuri_product_reco.loc[2, 'product_name']
       material3 = meijuri_product_reco.loc[2, 'material']
       price3 = meijuri_product_reco.loc[2, 'price_CAD']
       product_link3 = meijuri_product_reco.loc[0, 'product_link']
       
       return render_template("results.html",
                              user_image=x,
                              similar1=similar1,
                              label1=label1.title(),
                              product1=product1.title(),
                              price1=price1,
                              material1=material1,
                              product_link1=product_link1,
                              similar2=similar2,
                              label2=label2.title(),
                              product2=product2.title(),
                              price2=price2,
                              material2=material2,
                              product_link2=product_link2,
                              similar3=similar3,
                              label3=label3.title(),
                              product3=product3.title(),
                              price3=price3,
                              material3=material3,
                              product_link3=product_link3)
       
         
# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
