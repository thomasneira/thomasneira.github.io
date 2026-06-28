# Data Download Instructions

```bash
pip install kagglehub

python3 -c "
import kagglehub
path = kagglehub.dataset_download('mkechinov/ecommerce-events-history-in-electronics-store')
print('Downloaded to:', path)
"
```

Then copy `events.csv` from the download path to this directory as `ga4_events.csv`.

## Dataset Details

- **Source**: [eCommerce Events History in Electronics Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store)
- **Rows**: 885K events
- **Columns**: `event_time`, `event_type`, `product_id`, `category_id`, `category_code`, `brand`, `price`, `user_id`, `user_session`
- **Event types**: `view` (90%), `cart` (5%), `purchase` (4%)
