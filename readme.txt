{% extends 'main.html' %} {% block body_block %} {% for item in user_products %}
<div class="container mt-5">
  <div class="row">
    <div class="col-md-6">
      <img
        src="{{item.product.image.url}}"
        alt="Not Available"
        class="img-fluid product-img"
      />
    </div>
    <div class="col-md-6 product_descrpition">
      <h2>{{item.product.name}}</h2>
      <h3>Amount : Rs {{item.product.price}}</h3>
      <p>{{item.product.description }}</p>

      <div class="input-group mb-3">
        <p>Quantity :</p>
        <button class="btn btn-outline-secondary" id="decrementbtn">-</button>
        <input
          type="number"
          class="form-control text-center"
          value="{{item.cart_count }}"
          min="1"
          id="inputBtn"
        />
        <button class="btn btn-outline-secondary" id="incremntBtn">+</button>
      </div>
      <a href="{% url 'removeCart' id=item.id %}" class="btn btn-primary"
        >Remove</a
      >
    </div>
  </div>
</div>

<script>
  document.addEventListener("DOMContentLoaded", function () {
    const decremntBtn = document.getElementById("decrementbtn");
    const incremntBtn = document.getElementById("incremntBtn");
    const inputBtn = document.getElementById("inputBtn");

    decremntBtn.addEventListener("click", function () {
      UpdateQantity(-1);
    });

    incremntBtn.addEventListener("click", function () {
      UpdateQantity(1);
    });

    function UpdateQantity(change) {
      let currentvalue = parseInt(inputBtn.value);
      let newvalue = currentvalue + change;

      if (newvalue < 1) {
        newvalue = 1;
      }
      inputBtn.value = newvalue;
    }
  });
</script>
{% endfor %} {% endblock %}