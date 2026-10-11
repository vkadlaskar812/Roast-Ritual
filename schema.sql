-- Reference only: app_agent.py creates these tables itself via SQLAlchemy.
CREATE TABLE IF NOT EXISTS orders (
  id          SERIAL PRIMARY KEY,
  customer    VARCHAR(60) NOT NULL,
  total_inr   DOUBLE PRECISION NOT NULL,
  roast       TEXT,
  roast_level VARCHAR(10),
  created_at  TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders (customer);

CREATE TABLE IF NOT EXISTS order_items (
  id       SERIAL PRIMARY KEY,
  order_id INT NOT NULL REFERENCES orders(id),
  item     VARCHAR(80) NOT NULL
);

CREATE TABLE IF NOT EXISTS order_wit (
  order_id     INT PRIMARY KEY REFERENCES orders(id),
  amusement    INT NOT NULL,
  discount_pct INT NOT NULL
);

-- Ideas for the data-engineer flourish
-- Regulars and their favourite item:
SELECT o.customer, i.item, COUNT(*) AS n,
       RANK() OVER (PARTITION BY o.customer ORDER BY COUNT(*) DESC) AS rk
FROM orders o JOIN order_items i ON i.order_id = o.id
GROUP BY o.customer, i.item;

-- Busiest hours (IST):
SELECT EXTRACT(HOUR FROM created_at AT TIME ZONE 'Asia/Kolkata') AS hr, COUNT(*) AS orders
FROM orders GROUP BY hr ORDER BY orders DESC;
