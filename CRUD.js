// const express = require('express');
// const bodyParser = require('body-parser');
// const { MongoClient, ObjectId } = require('mongodb');
// const cors = require('cors');

// const app = express();
// const port = 3000;

// app.use(cors());  // Allow cross-origin requests from your frontend
// app.use(bodyParser.urlencoded({ extended: true }));
// app.use(bodyParser.json());

// // Menu Item Prices
// const menu = {
//   pizza: 200,
//   burger: 100,
//   pasta: 150,
//   coke: 50,
// };

// // MongoDB Connection URL (for Compass: default localhost)
// const url = 'mongodb://127.0.0.1:27017';
// const client = new MongoClient(url);
// const dbName = 'foodDeliveryDB';
// let db;

// // Connect to MongoDB
// async function connectMongo() {
//   await client.connect();
//   console.log('✅ Connected to MongoDB Compass');
//   db = client.db(dbName);
// }
// connectMongo().catch(console.error);

// // ✅ Create Order (POST)
// app.post('/placeOrder', async (req, res) => {
//   const name = req.body.name || 'Customer';
//   const items = req.body.items ? req.body.items.split(',') : [];

//   if (!name || items.length === 0) {
//     return res.status(400).send({ message: 'Name and items are required' });
//   }

//   let totalCost = 0;
//   items.forEach(item => {
//     if (menu[item]) totalCost += menu[item];
//   });

//   const orderCollection = db.collection('orders');
//   const result = await orderCollection.insertOne({
//     name,
//     items,
//     totalCost,
//     method: 'POST',
//     createdAt: new Date()
//   });

//   res.send({ message: 'Order placed successfully!', orderId: result.insertedId });
// });

// // ✅ Read All Orders (GET)
// app.get('/orders', async (req, res) => {
//   const orderCollection = db.collection('orders');
//   const orders = await orderCollection.find({}).toArray();
//   res.send(orders);
// });

// // ✅ Read Single Order by ID (GET)
// app.get('/orders/:id', async (req, res) => {
//   const orderId = req.params.id;
//   const orderCollection = db.collection('orders');

//   try {
//     const order = await orderCollection.findOne({ _id: new ObjectId(orderId) });
//     if (!order) return res.status(404).send({ message: 'Order not found' });
//     res.send(order);
//   } catch (error) {
//     res.status(400).send({ message: 'Invalid order ID format' });
//   }
// });

// // ✅ Update Order by ID (PUT)
// app.put('/orders/:id', async (req, res) => {
//   const orderId = req.params.id;
//   const { name, items } = req.body;

//   if (!name || !items) {
//     return res.status(400).send({ message: 'Name and items are required' });
//   }

//   const itemList = items.split(',');
//   let totalCost = 0;
  
//   itemList.forEach(item => {
//     if (menu[item]) totalCost += menu[item];
//   });
  
//   const orderCollection = db.collection('orders');
//   try {
//     const result = await orderCollection.updateOne(
//         { _id: new ObjectId(orderId) },
//         {
//           $set: {
//             name,
//             items: itemList,
//             totalCost,
//             updatedAt: new Date()
//           }
//         }
//       );
      

//     if (result.matchedCount === 0) return res.status(404).send({ message: 'Order not found' });
//     res.send({ message: 'Order updated successfully!' });

//   } catch (error) {
//     res.status(400).send({ message: 'Invalid order ID format' });
//   }
// });

// // ✅ Delete Order by ID (DELETE)
// app.delete('/orders/:id', async (req, res) => {
//   const orderId = req.params.id;
//   const orderCollection = db.collection('orders');

//   try {
//     const result = await orderCollection.deleteOne({ _id: new ObjectId(orderId) });
//     if (result.deletedCount === 0) return res.status(404).send({ message: 'Order not found' });
//     res.send({ message: 'Order deleted successfully!' });

//   } catch (error) {
//     res.status(400).send({ message: 'Invalid order ID format' });
//   }
// });

// app.listen(port, () => {
//   console.log(`🚀 Food Delivery Server running at http://localhost:${port}`);
// });
const express = require('express');
const bodyParser = require('body-parser');
const { MongoClient, ObjectId } = require('mongodb');
const cors = require('cors');

const app = express();
const port = 3000;

// Middlewares
app.use(cors());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

// Menu Item Prices
const menu = {
  pizza: 200,
  burger: 100,
  pasta: 150,
  coke: 50,
};

// MongoDB connection URI
const url = 'mongodb://127.0.0.1:27017';
const client = new MongoClient(url);
const dbName = 'foodDeliveryDB';
let db;

// Connect to MongoDB
async function connectMongo() {
  try {
    await client.connect();
    console.log('✅ Connected to MongoDB Compass');
    db = client.db(dbName);
  } catch (err) {
    console.error('❌ MongoDB connection error:', err);
  }
}
connectMongo();

// ====================== ROUTES ======================

// Create new order
app.post('/placeOrder', async (req, res) => {
  const name = req.body.name || 'Customer';
  const items = req.body.items ? req.body.items.split(',') : [];

  if (!name || items.length === 0) {
    return res.status(400).send({ message: 'Name and at least one item required' });
  }

  let totalCost = 0;
  items.forEach(item => {
    if (menu[item]) totalCost += menu[item];
  });

  const orderCollection = db.collection('orders');
  const result = await orderCollection.insertOne({
    name,
    items,
    totalCost,
    method: 'POST',
    createdAt: new Date()
  });

  res.send({ message: 'Order placed successfully!', orderId: result.insertedId });
});

// Fetch all orders
app.get('/orders', async (req, res) => {
  const orderCollection = db.collection('orders');
  const orders = await orderCollection.find({}).toArray();
  res.send(orders);
});

// Fetch single order by ID
app.get('/orders/:id', async (req, res) => {
  const orderId = req.params.id;
  const orderCollection = db.collection('orders');

  try {
    const order = await orderCollection.findOne({ _id: new ObjectId(orderId) });
    if (!order) return res.status(404).send({ message: 'Order not found' });
    res.send(order);
  } catch (error) {
    res.status(400).send({ message: 'Invalid order ID format' });
  }
});

// Update existing order (by _id — no duplicates)
app.put('/orders/:id', async (req, res) => {
  const orderId = req.params.id;
  const { name, items } = req.body;

  if (!name || !items) {
    return res.status(400).send({ message: 'Name and items required' });
  }

  const itemList = items.split(',');
  let totalCost = 0;

  itemList.forEach(item => {
    if (menu[item]) totalCost += menu[item];
  });

  const orderCollection = db.collection('orders');
  try {
    const result = await orderCollection.updateOne(
      { _id: new ObjectId(orderId) },
      {
        $set: {
          name,
          items: itemList,
          totalCost,
          updatedAt: new Date()
        }
      }
    );

    if (result.matchedCount === 0) return res.status(404).send({ message: 'Order not found' });
    res.send({ message: 'Order updated successfully!' });

  } catch (error) {
    res.status(400).send({ message: 'Invalid order ID format' });
  }
});

// Delete an order by ID
app.delete('/orders/:id', async (req, res) => {
  const orderId = req.params.id;
  const orderCollection = db.collection('orders');

  try {
    const result = await orderCollection.deleteOne({ _id: new ObjectId(orderId) });
    if (result.deletedCount === 0) return res.status(404).send({ message: 'Order not found' });
    res.send({ message: 'Order deleted successfully!' });
  } catch (error) {
    res.status(400).send({ message: 'Invalid order ID format' });
  }
});

// Server start
app.listen(port, () => {
  console.log(`🚀 Food Delivery Server running at http://localhost:${port}`);
});
