from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import numpy as np
from option_game import Call, Put, RiskReversal, CallSpread, PutSpread, Straddle, Strangle, Portfolio, DiceSimulator

app = Flask(__name__)
app.secret_key = 'dice_options_game_secret_key'

# Remove the global simulator
# Instead, we'll create a simulator per session

def get_simulator():
    """Get or create a simulator for the current session"""
    if 'simulator' not in session:
        # Initialize a new simulator for this session
        session['simulator'] = DiceSimulator().to_dict()
    
    # Convert the dictionary back to a DiceSimulator object
    simulator_dict = session['simulator']
    simulator = DiceSimulator.from_dict(simulator_dict)
    return simulator

def save_simulator(simulator):
    """Save the simulator state to the session"""
    session['simulator'] = simulator.to_dict()

@app.route('/')
def index():
    """Main page showing portfolio and game status"""
    # Get the simulator for this session
    simulator = get_simulator()
    
    # Calculate portfolio value
    portfolio_value = simulator.calculate_portfolio_value()
    
    # Calculate PNL if we have completed two rolls
    total_pnl, position_pnls = simulator.calculate_portfolio_pnl() if len(simulator.rolls) >= 2 else (None, [])
    
    # Get first_roll if available
    first_roll = simulator.rolls[0] if len(simulator.rolls) == 1 else None
    
    # Calculate updated deltas for all positions in the portfolio
    position_deltas = []
    for position, quantity, _ in simulator.portfolio.positions:
        position_deltas.append({
            'position': str(position),
            'delta': position.delta(first_roll),
            'quantity': quantity,
            'position_delta': position.delta(first_roll) * quantity
        })
    
    portfolio_delta = simulator.portfolio.delta(first_roll) if simulator.portfolio.positions else 0
    
    return render_template('index.html', 
                          simulator=simulator,
                          portfolio_value=portfolio_value,
                          rolls=simulator.rolls,
                          total=simulator.get_total() if simulator.rolls else 0,
                          total_pnl=total_pnl,
                          position_pnls=position_pnls,
                          position_deltas=position_deltas,
                          portfolio_delta=portfolio_delta,
                          first_roll=first_roll,
                          Call=Call,
                          Put=Put,
                          RiskReversal=RiskReversal,
                          CallSpread=CallSpread,
                          PutSpread=PutSpread,
                          Straddle=Straddle,
                          Strangle=Strangle)

@app.route('/add_option', methods=['POST'])
def add_option():
    """Add an option to the portfolio"""
    # Get the simulator for this session
    simulator = get_simulator()
    
    option_type = request.form.get('option_type')
    quantity = int(request.form.get('quantity', 1))
    entry_price = request.form.get('entry_price')
    
    # Convert entry_price to float if provided
    if entry_price and entry_price.strip():
        entry_price = float(entry_price)
    else:
        entry_price = None
    
    try:
        if option_type == 'call':
            strike = int(request.form.get('strike'))
            option = Call(strike)
        elif option_type == 'put':
            strike = int(request.form.get('strike'))
            option = Put(strike)
        elif option_type == 'risk_reversal':
            put_strike = int(request.form.get('put_strike'))
            call_strike = int(request.form.get('call_strike'))
            option = RiskReversal(put_strike, call_strike)
        elif option_type == 'call_spread':
            lower_strike = int(request.form.get('lower_strike'))
            higher_strike = int(request.form.get('higher_strike'))
            option = CallSpread(lower_strike, higher_strike)
        elif option_type == 'put_spread':
            higher_strike = int(request.form.get('higher_strike'))
            lower_strike = int(request.form.get('lower_strike'))
            option = PutSpread(higher_strike, lower_strike)
        elif option_type == 'straddle':
            strike = int(request.form.get('strike'))
            option = Straddle(strike)
        elif option_type == 'strangle':
            put_strike = int(request.form.get('put_strike'))
            call_strike = int(request.form.get('call_strike'))
            option = Strangle(put_strike, call_strike)
        else:
            flash('Invalid option type', 'error')
            return redirect(url_for('index'))
            
        simulator.add_to_portfolio(option, quantity, entry_price)
        
        # Save the updated simulator state
        save_simulator(simulator)
        
        price_info = f" at price {entry_price:.2f}" if entry_price is not None else ""
        flash(f'Added {quantity} x {option}{price_info} to portfolio', 'success')
        
    except ValueError as e:
        flash(f'Error: {str(e)}', 'error')
        
    return redirect(url_for('index'))

@app.route('/roll_die', methods=['POST'])
def roll_die():
    """Roll a single die"""
    # Get the simulator for this session
    simulator = get_simulator()
    
    value = request.form.get('value')
    if value:
        value = int(value)
    else:
        value = None
        
    simulator.roll_die(value)
    
    # If we've rolled two dice, calculate PNL
    position_pnls = []
    total_pnl = None
    if len(simulator.rolls) == 2:
        total_pnl, position_pnls = simulator.calculate_portfolio_pnl()
    
    # Save the updated simulator state
    save_simulator(simulator)
    
    flash(f'Rolled a {simulator.rolls[-1]}', 'success')
    return redirect(url_for('index'))

@app.route('/reset', methods=['POST'])
def reset():
    """Reset the simulator"""
    # Get the simulator for this session
    simulator = get_simulator()
    
    simulator.reset_rolls()
    # Create a new portfolio
    simulator.portfolio = Portfolio()
    
    # Save the updated simulator state
    save_simulator(simulator)
    
    flash('Game reset', 'success')
    return redirect(url_for('index'))

@app.route('/option_analytics', methods=['POST'])
def option_analytics():
    """Get analytics for an option before adding to portfolio"""
    # Get the simulator for this session
    simulator = get_simulator()
    
    option_type = request.form.get('option_type')
    quantity = int(request.form.get('quantity', 1))  # Default to 1 if not provided
    
    try:
        # Create the option object based on form data
        if option_type == 'call':
            strike = int(request.form.get('strike'))
            option = Call(strike)
        elif option_type == 'put':
            strike = int(request.form.get('strike'))
            option = Put(strike)
        elif option_type == 'risk_reversal':
            put_strike = int(request.form.get('put_strike'))
            call_strike = int(request.form.get('call_strike'))
            option = RiskReversal(put_strike, call_strike)
        elif option_type == 'call_spread':
            lower_strike = int(request.form.get('lower_strike'))
            higher_strike = int(request.form.get('higher_strike'))
            option = CallSpread(lower_strike, higher_strike)
        elif option_type == 'put_spread':
            higher_strike = int(request.form.get('higher_strike'))
            lower_strike = int(request.form.get('lower_strike'))
            option = PutSpread(higher_strike, lower_strike)
        elif option_type == 'straddle':
            strike = int(request.form.get('strike'))
            option = Straddle(strike)
        elif option_type == 'strangle':
            put_strike = int(request.form.get('put_strike'))
            call_strike = int(request.form.get('call_strike'))
            option = Strangle(put_strike, call_strike)
        else:
            return jsonify({'error': 'Invalid option type'}), 400
        
        # Calculate analytics with quantity
        analytics = simulator.get_option_analytics(option, quantity)
        
        # Return the analytics including per-contract and total values
        return jsonify({
            'option_name': str(option),
            'fair_value_per': round(analytics['fair_value_per'], 4),
            'fair_value_total': round(analytics['fair_value_total'], 4),
            'bid_per': round(analytics['bid_per'], 4),
            'ask_per': round(analytics['ask_per'], 4),
            'bid_total': round(analytics['bid_total'], 4),
            'ask_total': round(analytics['ask_total'], 4),
            'delta': round(analytics['delta_per_contract'], 4),
            'vega': round(analytics['vega_per_contract'], 4),
            'total_delta': round(analytics['total_delta'], 4),
            'total_vega': round(analytics['total_vega'], 4),
            'delta_neutral': round(analytics['delta_neutral_quantity'], 2)
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/remove_option/<int:index>', methods=['POST'])
def remove_option(index):
    """Remove an option from the portfolio by index"""
    # Get the simulator for this session
    simulator = get_simulator()
    
    # Try to remove the position
    removed = simulator.portfolio.remove_position(index)
    
    if removed:
        position, quantity, entry_price = removed
        # Save the updated simulator state
        save_simulator(simulator)
        flash(f'Removed {quantity} x {position} from portfolio', 'success')
    else:
        flash('Invalid position index', 'error')
        
    return redirect(url_for('index'))

@app.route('/payoff_curve', methods=['POST'])
def payoff_curve():
    simulator = get_simulator()
    option_type = request.form.get('option_type')
    quantity = int(request.form.get('quantity', 1))

    try:
        if option_type == 'call':
            strike = int(request.form.get('strike'))
            option = Call(strike)
        elif option_type == 'put':
            strike = int(request.form.get('strike'))
            option = Put(strike)
        elif option_type == 'risk_reversal':
            put_strike = int(request.form.get('put_strike'))
            call_strike = int(request.form.get('call_strike'))
            option = RiskReversal(put_strike, call_strike)
        elif option_type == 'call_spread':
            lower_strike = int(request.form.get('lower_strike'))
            higher_strike = int(request.form.get('higher_strike'))
            option = CallSpread(lower_strike, higher_strike)
        elif option_type == 'put_spread':
            higher_strike = int(request.form.get('higher_strike'))
            lower_strike = int(request.form.get('lower_strike'))
            option = PutSpread(higher_strike, lower_strike)
        elif option_type == 'straddle':
            strike = int(request.form.get('strike'))
            option = Straddle(strike)
        elif option_type == 'strangle':
            put_strike = int(request.form.get('put_strike'))
            call_strike = int(request.form.get('call_strike'))
            option = Strangle(put_strike, call_strike)
        else:
            return jsonify({'error': 'Invalid option type'}), 400

        curve = simulator.get_option_payoff_curve(option, quantity)
        return jsonify(curve)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5001)
