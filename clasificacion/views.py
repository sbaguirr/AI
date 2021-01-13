from django.shortcuts import render


# Create your views here.
def index(request):
    return render(request, "clasificacion/index.html")


def hashtag(request):
    if request.method == "POST":
        hashtag = request.POST["hashtag"]
        cantidad = request.POST["cantidad"]
        mesDesde = request.POST["mesDesde"]
        diaDesde = request.POST["diaDesde"]
        anioDesde = request.POST["anioDesde"]
        mesHasta = request.POST["mesHasta"]
        diaHasta = request.POST["diaHasta"]
        anioHasta = request.POST["anioHasta"]
        pedidosAyuda = []
        quejas = []
        ofertas = []
        pedidosAyuda.append(hashtag)
        pedidosAyuda.append(cantidad)
        pedidosAyuda.append(mesDesde)
        quejas.append(diaDesde)
        quejas.append(anioDesde)
        ofertas.append(mesHasta)
        ofertas.append(diaHasta)
        ofertas.append(anioHasta)
        return render(request, "clasificacion/hashtag.html", {
            "pedidosAyuda": pedidosAyuda,
            "quejas": quejas,
            "ofertas": ofertas
        })
    else:
        return render(request, "clasificacion/hashtag.html")


def tweet(request):
    return render(request, "clasificacion/tweet.html")
