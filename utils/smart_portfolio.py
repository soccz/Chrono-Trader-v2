import json
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
from utils.logger import logger
from utils.config import config

class Position:
    def __init__(self, market, entry_price=0, quantity=0):
        self.market = market
        self.entry_price = float(entry_price)
        self.quantity = float(quantity)
        self.max_price = entry_price # For trailing stop
        self.opened_at = datetime.now()
        self.last_updated = datetime.now()

    @property
    def value(self, current_price=None):
        price = current_price if current_price else self.entry_price
        return self.quantity * price

    @property
    def pnl_pct(self, current_price):
        if self.entry_price == 0: return 0.0
        return ((current_price - self.entry_price) / self.entry_price) * 100

    def add(self, price, qty):
        """Scale in (increase position)"""
        total_value = (self.quantity * self.entry_price) + (qty * price)
        self.quantity += qty
        self.entry_price = total_value / self.quantity if self.quantity > 0 else 0
        self.last_updated = datetime.now()

    def reduce(self, qty):
        """Scale out (decrease position)"""
        self.quantity = max(0, self.quantity - qty)
        self.last_updated = datetime.now()
        if self.quantity == 0:
            self.entry_price = 0

    def update_trailing(self, current_price):
        self.max_price = max(self.max_price, current_price)

    def to_dict(self):
        return {
            "market": self.market,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "max_price": self.max_price,
            "opened_at": self.opened_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }

    @staticmethod
    def from_dict(data):
        pos = Position(data['market'], data['entry_price'], data['quantity'])
        pos.max_price = data.get('max_price', data['entry_price'])
        if 'opened_at' in data:
            pos.opened_at = datetime.fromisoformat(data['opened_at'])
        return pos


class SmartPortfolioManager:
    """
    Manages portfolio state and executes Hybrid-Native Rebalancing.
    """
    def __init__(self, db_path="portfolio_state.json", initial_capital=1000000.0):
        self.db_path = db_path
        self.positions = {} # market -> Position
        self.cash = initial_capital
        self.total_equity_history = []
        self.load_state()

    def get_equity(self, current_prices: dict):
        equity = self.cash
        for market, pos in self.positions.items():
            price = current_prices.get(market, pos.entry_price)
            equity += pos.quantity * price
        return equity

    def sync_target_weight(self, market, target_weight, current_price, 
                          gate_value=0.5, consensus_score=0.5, uncertainty=1.0):
        """
        Calculates and executes rebalancing action based on target weight and Hybrid metrics.
        
        Args:
            market (str): Market ticker
            target_weight (float): Target portfolio % (0.0 to 1.0)
            current_price (float): Current market price
            gate_value (float): Trend(>0.6) vs Pattern(<0.4) indicator
            consensus_score (float): Ensemble agreement (0.0 to 1.0)
        
        Returns:
            dict: Action report {action, reason, quantity, filled_price}
        """
        # 1. Update Portfolio Valuation
        # Ideally we pass all current prices, but for single sync we estimate
        total_equity = self.get_equity({market: current_price})
        
        # 2. Hybrid Strategy Adjustment (The Brain)
        # Adjust target weight based on Consensus & Gate
        adjusted_weight = target_weight
        
        regime = "Hybrid"
        if gate_value > 0.6: regime = "Trend"
        elif gate_value < 0.4: regime = "Pattern"
        
        # Consensus Bonus: If agreement is high, scale up aggression
        if consensus_score >= 0.8 and regime == "Trend":
            adjusted_weight *= 1.2 # Pyramid
        elif consensus_score < 0.5:
            adjusted_weight *= 0.5 # Conservative
            
        # Hard Cap
        adjusted_weight = min(adjusted_weight, 0.5) # Max 50% per asset
        
        # 3. Calculate Deltas
        target_value = total_equity * adjusted_weight
        current_pos = self.positions.get(market, Position(market))
        current_value = current_pos.quantity * current_price
        
        diff_value = target_value - current_value
        min_trade_val = 5000 # KRW min trade size
        
        action_report = {"action": "HOLD", "reason": "Balanced", "market": market}
        
        # 4. Determine Action
        if abs(diff_value) < min_trade_val:
            return action_report
            
        if diff_value > 0:
            # BUY Logic
            buy_qty = diff_value / current_price
            if self.cash >= diff_value:
                self.cash -= diff_value
                current_pos.add(current_price, buy_qty)
                self.positions[market] = current_pos
                action_report = {
                    "action": "BUY", 
                    "quantity": buy_qty, 
                    "price": current_price,
                    "reason": f"Rebalance (Target {adjusted_weight*100:.1f}%) [{regime} + Consensus {consensus_score:.1f}]"
                }
            else:
                 action_report["reason"] = "Insufficient Cash"
                 
        elif diff_value < 0:
            # SELL Logic
            sell_qty = abs(diff_value) / current_price
            # Check pattern mode stop logic if needed
            self.cash += abs(diff_value)
            current_pos.reduce(sell_qty)
            
            if current_pos.quantity < 1e-8:
                del self.positions[market]
            else:
                self.positions[market] = current_pos
                
            action_report = {
                "action": "SELL", 
                "quantity": sell_qty, 
                "price": current_price, 
                "reason": f"Rebalance (Target {adjusted_weight*100:.1f}%)"
            }
            
        self.save_state()
        return action_report

    def load_state(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.cash = data.get('cash', 1000000.0)
                    pos_data = data.get('positions', {})
                    self.positions = {k: Position.from_dict(v) for k, v in pos_data.items()}
            except Exception as e:
                logger.error(f"Failed to load portfolio state: {e}")

    def save_state(self):
        try:
            data = {
                'cash': self.cash,
                'positions': {k: v.to_dict() for k, v in self.positions.items()},
                'updated_at': datetime.now().isoformat()
            }
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save portfolio state: {e}")
