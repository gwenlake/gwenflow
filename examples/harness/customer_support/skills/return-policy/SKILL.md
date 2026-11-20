---
name: return-policy
description: How to handle a return request, step by step.
---
1. Check the order with `order_status`: it must be `delivered`, less than 30 days ago.
2. Ask the reason if the customer did not give one.
3. Call `start_return` and give the customer the return label number.
4. Remind them: bike unused, original box, refund within 14 days of reception.
