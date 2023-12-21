import dill
from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
import pandas as pd

sched = BlockingScheduler(timezone=tzlocal.get_localzone_name())

df = pd.read_csv('data/train_data.csv')
with open('price_category_pipeline', 'rb') as file:
    model = dill.load(file)


@sched.scheduled_job('cron', second='*/5')
def on_time():
    data = df.sample(n=5)
    data['predicted_price_cat'] = model['model'].predict(data)
    print(data[['id', 'price', 'predicted_price_cat']])


if __name__ == '__main__':
    sched.start()
