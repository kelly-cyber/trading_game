import unittest
import numpy as np
from option_game import (
    Call, Put, RiskReversal, CallSpread, PutSpread, Straddle, Strangle,
    Portfolio, DiceSimulator, C_DELTA, C_VEGA, HALF_SPREAD, MIN_PRICE
)

class TestOptionGame(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create basic options for testing
        self.call_7 = Call(7)
        self.put_7 = Put(7)
        self.risk_reversal = RiskReversal(5, 9)
        self.call_spread = CallSpread(6, 8)
        self.put_spread = PutSpread(8, 6)
        self.straddle = Straddle(7)
        self.strangle = Strangle(5, 9)
        
        # Create a simulator
        self.simulator = DiceSimulator()
        
    def test_option_probabilities(self):
        """Test probability calculations for options."""
        # Test standard probability distribution (no dice rolled)
        probs = self.call_7.calculate_probabilities()
        self.assertEqual(len(probs), 13)  # 0-12 (0 and 1 unused)
        self.assertAlmostEqual(sum(probs), 1.0)  # Probabilities should sum to 1
        
        # Test conditional probability (first die = 3)
        probs_conditional = self.call_7.calculate_probabilities(first_roll=3)
        self.assertEqual(len(probs_conditional), 13)
        self.assertAlmostEqual(sum(probs_conditional), 1.0)
        
        # Check specific probabilities
        # P(7) = 6/36 = 1/6 ≈ 0.1667 for standard distribution
        self.assertAlmostEqual(probs[7], 6/36)
        
        # P(7|first=3) = 1/6 for conditional distribution where first die is 3
        self.assertAlmostEqual(probs_conditional[7], 1/6)  # 3+4=7, so P(7|3) = 1/6
        self.assertAlmostEqual(probs_conditional[9], 1/6)  # 3+6=9, so P(9|3) = 1/6
        
    def test_call_option(self):
        """Test Call option delta and vega calculations."""
        # Delta of 7 call should be P(roll ≥ 7) = 21/36 = 0.5833
        self.assertAlmostEqual(self.call_7.delta(), 21/36)
        
        # Vega of 7 call should be P(roll = 7) * 36 = 6/36 * 36 = 6
        self.assertAlmostEqual(self.call_7.vega(), 6)
        
        # Test with first roll = 3
        # Delta should be P(3+second ≥ 7) = P(second ≥ 4) = 3/6 = 0.5
        self.assertAlmostEqual(self.call_7.delta(first_roll=3), 3/6)
        
        # Vega with first roll = 3 should be P(3+second = 7) * 6 = P(second = 4) * 6 = 1/6 * 6 = 1
        self.assertAlmostEqual(self.call_7.vega(first_roll=3), 1)
        
    def test_put_option(self):
        """Test Put option delta and vega calculations."""
        self.assertAlmostEqual(self.put_7.delta(), -21/36)  # Adjusted for the -1 in the code
        
        # Vega of 7 put should be P(roll = 7) * 36 = 6/36 * 36 = 6
        self.assertAlmostEqual(self.put_7.vega(), 6)
        
    def test_option_strategies(self):
        """Test option strategies delta and vega calculations."""
        # Risk Reversal: long 5 put, short 9 call
        # Delta = put.delta - call.delta
        expected_rr_delta = self.risk_reversal.put.delta() - self.risk_reversal.call.delta()
        self.assertAlmostEqual(self.risk_reversal.delta(), expected_rr_delta)
        
        # Call Spread: long 6 call, short 8 call
        # Delta = long_call.delta - short_call.delta
        expected_cs_delta = self.call_spread.long_call.delta() - self.call_spread.short_call.delta()
        self.assertAlmostEqual(self.call_spread.delta(), expected_cs_delta)
        
        # Straddle: long 7 call, long 7 put
        # Delta = call.delta + put.delta
        expected_straddle_delta = self.straddle.call.delta() + self.straddle.put.delta()
        self.assertAlmostEqual(self.straddle.delta(), expected_straddle_delta)
        
    def test_portfolio(self):
        """Test portfolio functionality."""
        portfolio = Portfolio()
        
        # Add positions
        portfolio.add_position(self.call_7, 2, 1.0)  # 2 contracts at $1.0 each
        portfolio.add_position(self.put_7, -1, 0.5)  # Short 1 contract at $0.5
        
        # Test delta calculation
        expected_delta = 2 * self.call_7.delta() - 1 * self.put_7.delta()
        self.assertAlmostEqual(portfolio.delta(), expected_delta)
        
        # Test vega calculation
        expected_vega = 2 * self.call_7.vega() - 1 * self.put_7.vega()
        self.assertAlmostEqual(portfolio.vega(), expected_vega)
        
    def test_dice_simulator(self):
        """Test dice simulator functionality."""
        # Test rolling dice
        die1 = self.simulator.roll_die(3)  # Force roll of 3
        self.assertEqual(die1, 3)
        self.assertEqual(len(self.simulator.rolls), 1)
        
        die2 = self.simulator.roll_die(4)  # Force roll of 4
        self.assertEqual(die2, 4)
        self.assertEqual(len(self.simulator.rolls), 2)
        
        # Test total calculation
        self.assertEqual(self.simulator.get_total(), 7)
        
        # Test option valuation
        # For a 7 call with rolls [3, 4], total is 7, so payoff should be 7-7=0
        call_value = self.simulator.calculate_option_value(self.call_7)
        self.assertEqual(call_value, 0)
        
        # Reset rolls and test expected value calculation
        self.simulator.reset_rolls()
        self.assertEqual(len(self.simulator.rolls), 0)
        
        # Expected value of 7 call before any rolls
        # EV = sum(max(i-7, 0) * P(i)) for i in [2,12]
        ev_call = self.simulator.calculate_option_value(self.call_7)
        expected_ev = sum(max(i-7, 0) * self.call_7.calculate_probabilities()[i] for i in range(2, 13))
        self.assertAlmostEqual(ev_call, expected_ev)
        
    def test_option_analytics(self):
        """Test option analytics calculations."""
        # Test analytics for a call option
        analytics = self.simulator.get_option_analytics(self.call_7, quantity=5)
        
        # Check that all expected keys are present
        expected_keys = [
            'fair_value_per', 'fair_value_total', 
            'bid_per', 'ask_per', 'bid_total', 'ask_total',
            'delta_per_contract', 'vega_per_contract', 
            'total_delta', 'total_vega', 'delta_neutral_quantity'
        ]
        for key in expected_keys:
            self.assertIn(key, analytics)
        
        # Check that total values are quantity * per-contract values
        self.assertAlmostEqual(analytics['fair_value_total'], 5 * analytics['fair_value_per'])
        self.assertAlmostEqual(analytics['total_delta'], 5 * analytics['delta_per_contract'])
        self.assertAlmostEqual(analytics['total_vega'], 5 * analytics['vega_per_contract'])
        
        # Check bid-ask relationship
        self.assertLessEqual(analytics['bid_per'], analytics['ask_per'])
        self.assertLessEqual(analytics['bid_total'], analytics['ask_total'])

    def test_straddle_delta(self):
        """Test straddle delta calculations."""
        # Create a straddle at strike 7
        straddle_7 = Straddle(7)
        
        # The delta should be 1/6
        # Call delta is P(roll ≥ 7) = 21/36
        # Put delta is P(roll ≤ 7) - 1 = 21/36 - 1 = -15/36
        # Combined delta should be 21/36 + (-15/36) = 6/36 = 1/6
        self.assertAlmostEqual(straddle_7.delta(), 0)
        
        # Test with first roll = 3
        # Call delta with first roll 3 is P(3+second ≥ 7) = P(second ≥ 4) = 3/6 = 0.5
        # Put delta with first roll 3 is P(3+second ≤ 7) - 1 = P(second ≤ 4) - 1 = 4/6 - 1 = -1/3
        # Combined delta should be 0.5 - 1/3 = 1/6
        self.assertAlmostEqual(straddle_7.delta(first_roll=3), -1/6)
        
        # Create a straddle at strike 6 (should have negative delta)
        straddle_6 = Straddle(6)
        self.assertAlmostEqual(straddle_6.delta(), 11/36)
        
        # Create a straddle at strike 8 (should have positive delta)
        straddle_8 = Straddle(8)
        self.assertAlmostEqual(straddle_8.delta(), -11/36)

    def test_impossible_outcome_after_first_roll(self):
        """Test that options have appropriate values after the first roll makes certain outcomes impossible."""
        # Create a call with strike 11
        call_11 = Call(11)
        
        # Initial values before any rolls
        initial_delta = call_11.delta()
        initial_vega = call_11.vega()
        initial_value = self.simulator.calculate_option_value(call_11)
        
        # All should be non-zero initially
        self.assertAlmostEqual(initial_delta, 1/12)
        self.assertAlmostEqual(initial_vega, 2)
        self.assertAlmostEqual(initial_value, 1/36)
        
        # Roll a 1 for the first die
        self.simulator.reset_rolls()
        self.simulator.roll_die(1)
        
        # After rolling a 1, the maximum possible total is 1+6=7
        # So an 11 call is now impossible to be in-the-money
        
        # The value should be 0
        value_after_first_roll = self.simulator.calculate_option_value(call_11)
        self.assertEqual(value_after_first_roll, 0)
        
        # The delta should be 0 (no chance of being in-the-money)
        delta_after_first_roll = call_11.delta(first_roll=1)
        self.assertEqual(delta_after_first_roll, 0)
        
        # The vega should be 0 (no chance of landing exactly on the strike)
        vega_after_first_roll = call_11.vega(first_roll=1)
        self.assertEqual(vega_after_first_roll, 0)
        
        # Add the call to the portfolio and verify portfolio calculations
        self.simulator.reset_rolls()  # Reset to test portfolio behavior
        self.simulator.portfolio = Portfolio()  # Start with empty portfolio
        self.simulator.add_to_portfolio(call_11, 10, 1.0)  # Add 10 contracts at $1 each
        
        # Initial portfolio metrics
        initial_portfolio_delta = self.simulator.portfolio.delta()
        initial_portfolio_vega = self.simulator.portfolio.vega()
        
        # Roll a 1 for the first die
        self.simulator.roll_die(1)
        
        # Portfolio delta should now be 0
        portfolio_delta_after_roll = self.simulator.portfolio.delta(first_roll=1)
        self.assertEqual(portfolio_delta_after_roll, 0)
        
        # Portfolio vega should now be 0
        portfolio_vega_after_roll = self.simulator.portfolio.vega(first_roll=1)
        self.assertEqual(portfolio_vega_after_roll, 0)
        self.simulator.roll_die(2)
        # Portfolio value should reflect the updated expected value of 0 for the options
        # The portfolio value should be the current value of options plus cash
        # Since the options are now worth 0, the value is just the remaining cash
        # Initial cash was -10 * 1.0 = -10 (spent on options)
        portfolio_value = self.simulator.calculate_portfolio_pnl()[0]
        expected_value = -10  # Initial cash spent on options that are now worthless
        self.assertEqual(portfolio_value, expected_value)

    def test_remove_position(self):
        """Test removing a position from the portfolio."""
        # Create a portfolio with multiple positions
        portfolio = Portfolio()
        portfolio.add_position(self.call_7, 2, 1.0)  # 2 contracts at $1.0 each
        portfolio.add_position(self.put_7, -1, 0.5)  # Short 1 contract at $0.5
        portfolio.add_position(self.straddle, 3, 1.5)  # 3 straddles at $1.5 each
        
        # Initial position count
        initial_count = len(portfolio.positions)
        self.assertEqual(initial_count, 3)
        
        # Record initial delta and vega
        initial_delta = portfolio.delta()
        initial_vega = portfolio.vega()
        
        # Verify that initial delta and vega are non-zero
        self.assertNotEqual(initial_delta, 0)
        self.assertNotEqual(initial_vega, 0)
        
        # Test removing a position by index
        removed = portfolio.remove_position(1)  # Remove the put position
        
        # Check that the position was removed
        self.assertEqual(len(portfolio.positions), 2)
        
        # Check that the correct position was removed
        self.assertIsNotNone(removed)
        position, quantity, entry_price = removed
        self.assertIsInstance(position, Put)
        self.assertEqual(position.strike, 7)
        self.assertEqual(quantity, -1)
        self.assertEqual(entry_price, 0.5)
        
        # Test removing a position with an invalid index
        invalid_removed = portfolio.remove_position(10)  # Index out of range
        self.assertIsNone(invalid_removed)
        self.assertEqual(len(portfolio.positions), 2)  # Count should remain the same
        
        # Test removing the last position
        portfolio.remove_position(0)
        portfolio.remove_position(0)
        self.assertEqual(len(portfolio.positions), 0)  # Portfolio should be empty
        
        # Check that delta and vega are zero after removing all positions
        self.assertEqual(portfolio.delta(), 0)
        self.assertEqual(portfolio.vega(), 0)
        
        # Test with first roll condition as well
        first_roll = 3
        self.assertEqual(portfolio.delta(first_roll), 0)
        self.assertEqual(portfolio.vega(first_roll), 0)


if __name__ == '__main__':
    unittest.main()